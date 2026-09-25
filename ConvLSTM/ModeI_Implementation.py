from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("processed")
MODEL_DIR = Path("models/convlstm_30min")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Temporal configuration
INPUT_STEPS = 6
HORIZON_STEPS = 6

# Target offset relative to the first input frame
#
# Inputs:
#   i, i+1, ..., i+5
#
# Target:
#   i+11
#
# Therefore the target is 6 five-minute steps
# (30 minutes) after the final input frame.
TARGET_OFFSET = INPUT_STEPS - 1 + HORIZON_STEPS

TIME_INTERVAL_MINUTES = 5

# Spatial dimensions
HEIGHT = 200
WIDTH = 200
CHANNELS = 1

# Dataset split by selected days
N_TRAIN_DAYS = 70
N_VAL_DAYS = 15
N_TEST_DAYS = 15

# Training configuration
BATCH_SIZE = 2
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3

# Normalization configuration
CLIP_PERCENTILE = 99.99
CLIP_SAMPLE_SIZE = 3000

# Reproducibility
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Load processed MeteoNet data
# ============================================================

radar = np.load(
    DATA_DIR / "radar_selected.npy",
    mmap_mode="r"
)

dates = np.load(
    DATA_DIR / "radar_dates_selected.npy"
)

lat = np.load(
    DATA_DIR / "lat_200.npy"
)

lon = np.load(
    DATA_DIR / "lon_200.npy"
)

top_100_days = pd.read_csv(
    DATA_DIR / "top_100_rainy_days.csv",
    parse_dates=["date"]
)

dates_pd = pd.DatetimeIndex(dates)


# ============================================================
# Check loaded data
# ============================================================

print("\nLoaded dataset")
print("------------------------")
print("Radar shape:", radar.shape)
print("Radar dtype:", radar.dtype)
print("Date shape:", dates.shape)
print("Latitude shape:", lat.shape)
print("Longitude shape:", lon.shape)
print(
    "Period:",
    dates_pd.min(),
    "to",
    dates_pd.max()
)


# Check radar dimensions
if radar.ndim != 3:
    raise ValueError(
        f"Expected radar data with 3 dimensions "
        f"(time, height, width), but got {radar.shape}."
    )

if radar.shape[1:] != (HEIGHT, WIDTH):
    raise ValueError(
        f"Expected radar spatial dimensions "
        f"{HEIGHT}x{WIDTH}, but got {radar.shape[1:]}."
    )

if len(radar) != len(dates):
    raise ValueError(
        f"Radar/date mismatch: {len(radar)} radar frames "
        f"and {len(dates)} timestamps."
    )

if lat.shape != (HEIGHT, WIDTH):
    raise ValueError(
        f"Expected latitude grid {(HEIGHT, WIDTH)}, "
        f"but got {lat.shape}."
    )

if lon.shape != (HEIGHT, WIDTH):
    raise ValueError(
        f"Expected longitude grid {(HEIGHT, WIDTH)}, "
        f"but got {lon.shape}."
    )


# ============================================================
# Identify temporally valid sequences
# ============================================================

# A sequence is valid only when every timestamp from the
# first input frame through the target frame is separated
# by exactly five minutes.

expected_delta = np.timedelta64(
    TIME_INTERVAL_MINUTES,
    "m"
)

valid_start_idx = []

for i in range(
    len(dates_pd) - TARGET_OFFSET
):

    sequence_dates = dates_pd[
        i:i + TARGET_OFFSET + 1
    ]

    diffs = np.diff(
        sequence_dates.values
    )

    if np.all(
        diffs == expected_delta
    ):
        valid_start_idx.append(i)

valid_start_idx = np.asarray(
    valid_start_idx,
    dtype=np.int32
)

if len(valid_start_idx) == 0:
    raise ValueError(
        "No temporally continuous radar sequences "
        "were found."
    )

print(
    "\nValid sequences:",
    len(valid_start_idx)
)


# ============================================================
# Check an example sequence
# ============================================================

example_idx = valid_start_idx[0]

print("\nExample sequence")
print("------------------------")

print("Inputs:")

for k in range(INPUT_STEPS):
    print(
        f"  Frame {k + 1}:",
        dates_pd[example_idx + k]
    )

example_target_idx = (
    example_idx + TARGET_OFFSET
)

print(
    "\nTarget:",
    dates_pd[example_target_idx]
)

print(
    "Forecast horizon from final input:",
    dates_pd[example_target_idx]
    - dates_pd[
        example_idx + INPUT_STEPS - 1
    ]
)


# ============================================================
# Chronological day-based split
# ============================================================

# Obtain selected days in chronological order.
unique_days = np.sort(
    np.unique(
        dates_pd.normalize()
    )
)

expected_days = (
    N_TRAIN_DAYS
    + N_VAL_DAYS
    + N_TEST_DAYS
)

if len(unique_days) != expected_days:
    raise ValueError(
        f"Expected {expected_days} selected days, "
        f"but found {len(unique_days)}."
    )

train_days = unique_days[
    :N_TRAIN_DAYS
]

val_days = unique_days[
    N_TRAIN_DAYS:
    N_TRAIN_DAYS + N_VAL_DAYS
]

test_days = unique_days[
    N_TRAIN_DAYS + N_VAL_DAYS:
    expected_days
]


# ============================================================
# Assign sequences to train, validation and test sets
# ============================================================

def select_sequences_for_days(
    valid_indices,
    allowed_days,
    dates_index,
    target_offset
):
    """
    Select sequences for which both the first input frame
    and target frame belong to the specified set of days.

    Parameters
    ----------
    valid_indices : array-like
        Valid sequence starting indices.

    allowed_days : array-like
        Days assigned to the dataset split.

    dates_index : pandas.DatetimeIndex
        Radar timestamps.

    target_offset : int
        Offset between the first input frame and target.

    Returns
    -------
    numpy.ndarray
        Selected sequence starting indices.
    """

    allowed_days = set(
        pd.Timestamp(day)
        for day in allowed_days
    )

    selected = []

    for i in valid_indices:

        input_day = (
            dates_index[i].normalize()
        )

        target_day = (
            dates_index[
                i + target_offset
            ].normalize()
        )

        if (
            input_day in allowed_days
            and target_day in allowed_days
        ):
            selected.append(i)

    return np.asarray(
        selected,
        dtype=np.int32
    )


train_idx = select_sequences_for_days(
    valid_indices=valid_start_idx,
    allowed_days=train_days,
    dates_index=dates_pd,
    target_offset=TARGET_OFFSET
)

val_idx = select_sequences_for_days(
    valid_indices=valid_start_idx,
    allowed_days=val_days,
    dates_index=dates_pd,
    target_offset=TARGET_OFFSET
)

test_idx = select_sequences_for_days(
    valid_indices=valid_start_idx,
    allowed_days=test_days,
    dates_index=dates_pd,
    target_offset=TARGET_OFFSET
)


# ============================================================
# Check dataset splits
# ============================================================

print("\nDataset split")
print("------------------------")

print(
    f"Training days:   {len(train_days)}"
)

print(
    f"Validation days: {len(val_days)}"
)

print(
    f"Testing days:    {len(test_days)}"
)

print()

print(
    f"Training sequences:   {len(train_idx)}"
)

print(
    f"Validation sequences: {len(val_idx)}"
)

print(
    f"Testing sequences:    {len(test_idx)}"
)


if len(train_idx) == 0:
    raise ValueError(
        "Training split contains no valid sequences."
    )

if len(val_idx) == 0:
    raise ValueError(
        "Validation split contains no valid sequences."
    )

if len(test_idx) == 0:
    raise ValueError(
        "Test split contains no valid sequences."
    )


# ============================================================
# Determine training frames used to estimate clipping value
# ============================================================

# Only training data are used to determine the clipping
# threshold to prevent information leakage from validation
# or test data.

train_frame_idx = []

for i in train_idx:

    # Input frames
    train_frame_idx.extend(
        range(
            i,
            i + INPUT_STEPS
        )
    )

    # Target frame
    train_frame_idx.append(
        i + TARGET_OFFSET
    )

train_frame_idx = np.unique(
    np.asarray(
        train_frame_idx,
        dtype=np.int32
    )
)

print(
    "\nUnique training radar frames:",
    len(train_frame_idx)
)


# ============================================================
# Estimate clipping value from training data
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

sample_size = min(
    CLIP_SAMPLE_SIZE,
    len(train_frame_idx)
)

sample_idx = rng.choice(
    train_frame_idx,
    size=sample_size,
    replace=False
)

sample_values = radar[
    sample_idx
]

clip_value = np.percentile(
    sample_values,
    CLIP_PERCENTILE
)

clip_value = np.float32(
    clip_value
)

del sample_values


if not np.isfinite(clip_value):
    raise ValueError(
        "The estimated clipping value is not finite."
    )

if clip_value <= 0:
    raise ValueError(
        f"Invalid clipping value: {clip_value}"
    )


print(
    f"Estimated {CLIP_PERCENTILE}th "
    f"percentile:",
    clip_value
)


# ============================================================
# Radar sequence generator
# ============================================================

class RadarSequence(keras.utils.Sequence):
    """
    Keras sequence generator for MeteoNet radar forecasting.

    Each sample contains INPUT_STEPS consecutive radar
    frames and one radar target TARGET_OFFSET frames after
    the first input frame.

    Normalization is performed on the fly using:

        x_clipped = clip(x, 0, clip_value)

        x_log = log(1 + x_clipped)

        x_normalized =
            x_log / log(1 + clip_value)

    This maps radar values to approximately [0, 1].
    """

    def __init__(
        self,
        radar,
        start_indices,
        clip_value,
        input_steps,
        target_offset,
        height,
        width,
        batch_size=2,
        shuffle=False,
        seed=42,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.radar = radar

        self.start_indices = np.asarray(
            start_indices,
            dtype=np.int32
        )

        self.clip_value = np.float32(
            clip_value
        )

        self.log_clip = np.log1p(
            self.clip_value
        )

        self.input_steps = input_steps
        self.target_offset = target_offset

        self.height = height
        self.width = width

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.rng = np.random.default_rng(
            seed
        )

        self.indices = np.arange(
            len(self.start_indices),
            dtype=np.int32
        )

        self.on_epoch_end()


    def __len__(self):
        """
        Number of batches per epoch.
        """

        return int(
            np.ceil(
                len(self.start_indices)
                / self.batch_size
            )
        )


    def normalize(self, x):
        """
        Clip and log-normalize radar values.
        """

        x = np.clip(
            x,
            0,
            self.clip_value
        )

        x = np.log1p(x)

        x = (
            x
            / self.log_clip
        )

        return x.astype(
            np.float32
        )


    def __getitem__(
        self,
        batch_index
    ):
        """
        Generate one batch.
        """

        start = (
            batch_index
            * self.batch_size
        )

        end = min(
            start + self.batch_size,
            len(self.indices)
        )

        batch_ids = self.indices[
            start:end
        ]

        batch_start_idx = (
            self.start_indices[
                batch_ids
            ]
        )

        current_batch_size = len(
            batch_start_idx
        )

        # ConvLSTM input:
        # batch x time x height x width x channel
        X = np.empty(
            (
                current_batch_size,
                self.input_steps,
                self.height,
                self.width,
                CHANNELS
            ),
            dtype=np.float32
        )

        # Target:
        # batch x height x width x channel
        y = np.empty(
            (
                current_batch_size,
                self.height,
                self.width,
                CHANNELS
            ),
            dtype=np.float32
        )

        for j, i in enumerate(
            batch_start_idx
        ):

            # Consecutive input radar frames
            input_frames = self.radar[
                i:i + self.input_steps
            ]

            # Forecast target
            target_frame = self.radar[
                i + self.target_offset
            ]

            # Normalize on the fly
            input_frames = self.normalize(
                input_frames
            )

            target_frame = self.normalize(
                target_frame
            )

            X[j, ..., 0] = input_frames

            y[j, ..., 0] = target_frame

        return X, y


    def on_epoch_end(self):
        """
        Shuffle training sequence order after each epoch.
        """

        if self.shuffle:
            self.rng.shuffle(
                self.indices
            )


# ============================================================
# Create generators
# ============================================================

train_gen = RadarSequence(
    radar=radar,
    start_indices=train_idx,
    clip_value=clip_value,
    input_steps=INPUT_STEPS,
    target_offset=TARGET_OFFSET,
    height=HEIGHT,
    width=WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)

val_gen = RadarSequence(
    radar=radar,
    start_indices=val_idx,
    clip_value=clip_value,
    input_steps=INPUT_STEPS,
    target_offset=TARGET_OFFSET,
    height=HEIGHT,
    width=WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=RANDOM_SEED
)

test_gen = RadarSequence(
    radar=radar,
    start_indices=test_idx,
    clip_value=clip_value,
    input_steps=INPUT_STEPS,
    target_offset=TARGET_OFFSET,
    height=HEIGHT,
    width=WIDTH,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=RANDOM_SEED
)


# ============================================================
# Check generator output
# ============================================================

X_sample, y_sample = train_gen[0]

print("\nGenerator check")
print("------------------------")

print(
    "Input shape:",
    X_sample.shape
)

print(
    "Target shape:",
    y_sample.shape
)

print(
    "Input range:",
    float(X_sample.min()),
    "to",
    float(X_sample.max())
)

print(
    "Target range:",
    float(y_sample.min()),
    "to",
    float(y_sample.max())
)


expected_X_shape = (
    X_sample.shape[0],
    INPUT_STEPS,
    HEIGHT,
    WIDTH,
    CHANNELS
)

expected_y_shape = (
    y_sample.shape[0],
    HEIGHT,
    WIDTH,
    CHANNELS
)

if X_sample.shape != expected_X_shape:
    raise ValueError(
        f"Unexpected input shape: "
        f"{X_sample.shape}"
    )

if y_sample.shape != expected_y_shape:
    raise ValueError(
        f"Unexpected target shape: "
        f"{y_sample.shape}"
    )

del X_sample
del y_sample


# ============================================================
# Build ConvLSTM model
# ============================================================

model = keras.Sequential(
    [
        keras.layers.Input(
            shape=(
                INPUT_STEPS,
                HEIGHT,
                WIDTH,
                CHANNELS
            )
        ),

        keras.layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True
        ),

        keras.layers.BatchNormalization(),

        keras.layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False
        ),

        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding="same",
            activation="sigmoid"
        )
    ],
    name="meteonet_convlstm_30min"
)

model.summary()


# ============================================================
# Compile model
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# Training callbacks
# ============================================================

best_model_path = (
    MODEL_DIR
    / "convlstm_meteonet_30min.keras"
)

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# Train model
# ============================================================

print("\nTraining model")
print("------------------------")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=MAX_EPOCHS,
    callbacks=callbacks
)


# ============================================================
# Save training history and experiment information
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    np.arange(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)


# Save clipping threshold
np.save(
    MODEL_DIR / "clip_value.npy",
    np.asarray(
        clip_value,
        dtype=np.float32
    )
)


# Save sequence indices
np.save(
    MODEL_DIR / "train_idx.npy",
    train_idx
)

np.save(
    MODEL_DIR / "val_idx.npy",
    val_idx
)

np.save(
    MODEL_DIR / "test_idx.npy",
    test_idx
)


# Save day splits
np.save(
    MODEL_DIR / "train_days.npy",
    train_days
)

np.save(
    MODEL_DIR / "val_days.npy",
    val_days
)

np.save(
    MODEL_DIR / "test_days.npy",
    test_days
)


# Save model configuration
config = pd.DataFrame(
    {
        "parameter": [
            "input_steps",
            "horizon_steps",
            "target_offset",
            "time_interval_minutes",
            "height",
            "width",
            "batch_size",
            "max_epochs",
            "initial_learning_rate",
            "clip_percentile",
            "clip_sample_size",
            "random_seed"
        ],

        "value": [
            INPUT_STEPS,
            HORIZON_STEPS,
            TARGET_OFFSET,
            TIME_INTERVAL_MINUTES,
            HEIGHT,
            WIDTH,
            BATCH_SIZE,
            MAX_EPOCHS,
            LEARNING_RATE,
            CLIP_PERCENTILE,
            CLIP_SAMPLE_SIZE,
            RANDOM_SEED
        ]
    }
)

config.to_csv(
    MODEL_DIR / "model_config.csv",
    index=False
)


print(
    "\nModel and training information saved to:",
    MODEL_DIR.resolve()
)


# ============================================================
# Reload best model
# ============================================================

# ModelCheckpoint saved the model with the lowest validation
# loss. Reload it explicitly so evaluation always uses the
# saved best-performing model.

model = keras.models.load_model(
    best_model_path
)


# ============================================================
# Inverse normalization
# ============================================================

def inverse_transform(
    x,
    clip_value
):
    """
    Transform normalized radar values back to the original
    rainfall scale used before normalization.

    Note that values above the training-derived clipping
    threshold cannot be recovered because they were clipped
    before normalization.
    """

    x = np.asarray(
        x,
        dtype=np.float32
    )

    return np.expm1(
        x
        * np.log1p(clip_value)
    )


# ============================================================
# Evaluate ConvLSTM and persistence baseline
# ============================================================

model_se = 0.0
model_ae = 0.0

persistence_se = 0.0
persistence_ae = 0.0

n_pixels = 0


print("\nEvaluating test set")
print("------------------------")

for batch_number in range(
    len(test_gen)
):

    X_batch, y_batch = (
        test_gen[batch_number]
    )

    # --------------------------------------------------------
    # ConvLSTM forecast
    # --------------------------------------------------------

    pred_batch = (
        model.predict_on_batch(
            X_batch
        )
    )


    # --------------------------------------------------------
    # Persistence forecast
    # --------------------------------------------------------

    # Persistence assumes that the latest observed radar
    # field remains unchanged over the 30-minute forecast
    # horizon.
    persistence_batch = (
        X_batch[:, -1]
    )


    # --------------------------------------------------------
    # Transform predictions back to rainfall units
    # --------------------------------------------------------

    y_true = inverse_transform(
        y_batch,
        clip_value
    )

    y_pred = inverse_transform(
        pred_batch,
        clip_value
    )

    y_persistence = inverse_transform(
        persistence_batch,
        clip_value
    )


    # --------------------------------------------------------
    # ConvLSTM errors
    # --------------------------------------------------------

    diff_model = (
        y_pred - y_true
    )

    model_se += np.sum(
        diff_model ** 2,
        dtype=np.float64
    )

    model_ae += np.sum(
        np.abs(diff_model),
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Persistence errors
    # --------------------------------------------------------

    diff_persistence = (
        y_persistence
        - y_true
    )

    persistence_se += np.sum(
        diff_persistence ** 2,
        dtype=np.float64
    )

    persistence_ae += np.sum(
        np.abs(
            diff_persistence
        ),
        dtype=np.float64
    )


    # Number of evaluated pixels
    n_pixels += y_true.size


    if (
        (batch_number + 1) % 500 == 0
        or batch_number + 1 == len(test_gen)
    ):
        print(
            f"Processed "
            f"{batch_number + 1} / "
            f"{len(test_gen)} batches"
        )


# ============================================================
# Calculate evaluation metrics
# ============================================================

model_mse = (
    model_se
    / n_pixels
)

model_rmse = np.sqrt(
    model_mse
)

model_mae = (
    model_ae
    / n_pixels
)


persistence_mse = (
    persistence_se
    / n_pixels
)

persistence_rmse = np.sqrt(
    persistence_mse
)

persistence_mae = (
    persistence_ae
    / n_pixels
)


# ============================================================
# Calculate percentage improvement over persistence
# ============================================================

mse_improvement = (
    (
        persistence_mse
        - model_mse
    )
    / persistence_mse
    * 100.0
)

rmse_improvement = (
    (
        persistence_rmse
        - model_rmse
    )
    / persistence_rmse
    * 100.0
)

mae_improvement = (
    (
        persistence_mae
        - model_mae
    )
    / persistence_mae
    * 100.0
)


# ============================================================
# Display results
# ============================================================

print("\nConvLSTM")
print("------------------------")
print("MSE :", model_mse)
print("RMSE:", model_rmse)
print("MAE :", model_mae)

print("\nPersistence")
print("------------------------")
print("MSE :", persistence_mse)
print("RMSE:", persistence_rmse)
print("MAE :", persistence_mae)

print("\nImprovement over persistence")
print("------------------------")
print(
    f"MSE improvement : "
    f"{mse_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)

print(
    f"MAE improvement : "
    f"{mae_improvement:.2f}%"
)


# ============================================================
# Save evaluation results
# ============================================================

evaluation_df = pd.DataFrame(
    {
        "metric": [
            "MSE",
            "RMSE",
            "MAE"
        ],

        "ConvLSTM": [
            model_mse,
            model_rmse,
            model_mae
        ],

        "Persistence": [
            persistence_mse,
            persistence_rmse,
            persistence_mae
        ],

        "Improvement_percent": [
            mse_improvement,
            rmse_improvement,
            mae_improvement
        ]
    }
)

evaluation_df.to_csv(
    MODEL_DIR / "test_metrics.csv",
    index=False
)


# ============================================================
# Final summary
# ============================================================

print("\nExperiment complete")
print("------------------------")

print(
    "Best model:",
    best_model_path
)

print(
    "Training history:",
    MODEL_DIR / "training_history.csv"
)

print(
    "Test metrics:",
    MODEL_DIR / "test_metrics.csv"
)

print(
    "Clip value:",
    MODEL_DIR / "clip_value.npy"
)

print(
    "Train/validation/test indices saved in:",
    MODEL_DIR
)
