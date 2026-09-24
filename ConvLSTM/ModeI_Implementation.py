import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# load data
data_dir = "data"

radar = np.load(
    f"{data_dir}/radar_selected.npy",
    mmap_mode="r"
)

dates = np.load(
    f"{data_dir}/radar_dates_selected.npy"
)

lat = np.load(
    f"{data_dir}/lat_200.npy"
)

lon = np.load(
    f"{data_dir}/lon_200.npy"
)

top_100_days = pd.read_csv(
    f"{data_dir}/top_100_rainy_days.csv",
    parse_dates=["date"]
)

import numpy as np
import pandas as pd

dates_pd = pd.DatetimeIndex(dates)

input_steps = 6
horizon_steps = 6   # 30 minutes / 5 minutes

target_offset = input_steps - 1 + horizon_steps
# = 11

valid_start_idx = []

for i in range(len(dates) - target_offset):

    # timestamps from first input through target
    sequence_dates = dates_pd[i:i + target_offset + 1]

    # Require every adjacent timestamp to be exactly 5 minutes apart
    diffs = np.diff(sequence_dates.values)

    if np.all(
        diffs == np.timedelta64(5, "m")
    ):
        valid_start_idx.append(i)

valid_start_idx = np.array(
    valid_start_idx,
    dtype=np.int32
)

i = valid_start_idx[0]

# Split data
# Unique selected days in chronological order
unique_days = np.sort(
    np.unique(dates_pd.normalize())
)


train_days = unique_days[:70]
val_days   = unique_days[70:85]
test_days  = unique_days[85:100]

train_idx = np.array([
    i for i in valid_start_idx
    if dates_pd[i].normalize() in train_days
    and dates_pd[i + 11].normalize() in train_days
], dtype=np.int32)

val_idx = np.array([
    i for i in valid_start_idx
    if dates_pd[i].normalize() in val_days
    and dates_pd[i + 11].normalize() in val_days
], dtype=np.int32)

test_idx = np.array([
    i for i in valid_start_idx
    if dates_pd[i].normalize() in test_days
    and dates_pd[i + 11].normalize() in test_days
], dtype=np.int32)

print("Training:", len(train_idx))
print("Validation:", len(val_idx))
print("Testing:", len(test_idx))

# train data
train_frame_idx = []

for i in train_idx:
    # 6 input frames
    train_frame_idx.extend(range(i, i + 6))

    # target frame
    train_frame_idx.append(i + 11)

train_frame_idx = np.unique(
    np.array(train_frame_idx, dtype=np.int32)
)

print("Unique training radar frames:", len(train_frame_idx))

# value to clip
rng = np.random.default_rng(42)

sample_size = min(
    3000,
    len(train_frame_idx)
)

sample_idx = rng.choice(
    train_frame_idx,
    size=sample_size,
    replace=False
)

sample_values = radar[sample_idx]

clip_value = np.percentile(
    sample_values,
    99.99
)

print("Estimated 99.99th percentile:", clip_value)

def normalize_radar(x, clip_value):
    x = np.clip(x, 0, clip_value)

    x = np.log1p(x)

    x = x / np.log1p(clip_value)

    return x.astype(np.float32)
  
# Create Sequence
class RadarSequence(keras.utils.Sequence):

    def __init__(
        self,
        radar,
        start_indices,
        clip_value,
        batch_size=2,
        shuffle=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.radar = radar
        self.start_indices = np.asarray(
            start_indices,
            dtype=np.int32
        )

        self.clip_value = np.float32(clip_value)
        self.log_clip = np.log1p(self.clip_value)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.start_indices)
        )

        self.on_epoch_end()

    def __len__(self):
        return int(
            np.ceil(
                len(self.start_indices)
                / self.batch_size
            )
        )

    def normalize(self, x):

        x = np.clip(
            x,
            0,
            self.clip_value
        )

        x = np.log1p(x)

        x = x / self.log_clip

        return x.astype(
            np.float32
        )

    def __getitem__(self, batch_index):

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
                6,
                200,
                200,
                1
            ),
            dtype=np.float32
        )

        # Target:
        # batch x height x width x channel
        y = np.empty(
            (
                current_batch_size,
                200,
                200,
                1
            ),
            dtype=np.float32
        )

        for j, i in enumerate(
            batch_start_idx
        ):

            # Six consecutive input radar frames
            input_frames = self.radar[
                i:i + 6
            ]

            # 30-minute-ahead target
            target_frame = self.radar[
                i + 11
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

        if self.shuffle:
            np.random.shuffle(
                self.indices
            )

# Create Batch
batch_size = 2

train_gen = RadarSequence(
    radar=radar,
    start_indices=train_idx,
    clip_value=clip_value,
    batch_size=batch_size,
    shuffle=True
)

val_gen = RadarSequence(
    radar=radar,
    start_indices=val_idx,
    clip_value=clip_value,
    batch_size=batch_size,
    shuffle=False
)

test_gen = RadarSequence(
    radar=radar,
    start_indices=test_idx,
    clip_value=clip_value,
    batch_size=batch_size,
    shuffle=False
)

#check input and target
i = train_idx[0]

print("Inputs:")

for k in range(6):
    print(
        k + 1,
        dates_pd[i + k]
    )

print(
    "\nTarget:",
    dates_pd[i + 11]
)

print(
    "\nHorizon from final input:",
    dates_pd[i + 11]
    - dates_pd[i + 5]
)

# model

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(6, 200, 200, 1)
    ),

    tf.keras.layers.ConvLSTM2D(
        32,
        (3, 3),
        padding="same",
        return_sequences=True
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ConvLSTM2D(
        16,
        (3, 3),
        padding="same",
        return_sequences=False
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        8,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid"
    )
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="mse",
    metrics=["mae"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),

    keras.callbacks.ModelCheckpoint(
        "convlstm_meteonet_30min.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# Save model

from pathlib import Path

model_dir = Path("models/convlstm_30min")
model_dir.mkdir(parents=True, exist_ok=True)

model.save(
    model_dir / "convlstm_meteonet_30min.keras"
)

history_df.to_csv(
    model_dir / "training_history.csv",
    index=False
)

np.save(
    model_dir / "clip_value.npy",
    np.array(clip_value, dtype=np.float32)
)

np.save(
    model_dir / "train_idx.npy",
    train_idx
)

np.save(
    model_dir / "val_idx.npy",
    val_idx
)

np.save(
    model_dir / "test_idx.npy",
    test_idx
)

print("Everything saved to:", model_dir)

# Calculate model accuracy

def inverse_transform(x, clip_value):

    x = np.asarray(x, dtype=np.float32)

    return np.expm1(
        x * np.log1p(clip_value)
    )

model_se = 0.0
model_ae = 0.0

persistence_se = 0.0
persistence_ae = 0.0

n_pixels = 0

for batch_number in range(len(test_gen)):

    X_batch, y_batch = test_gen[batch_number]

    # ConvLSTM prediction
    pred_batch = model.predict_on_batch(
        X_batch
    )

    # Persistence:
    # latest observed radar frame
    persistence_batch = X_batch[:, -1]

    # Back to original radar units
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

    # ConvLSTM errors
    diff_model = y_pred - y_true

    model_se += np.sum(
        diff_model ** 2,
        dtype=np.float64
    )

    model_ae += np.sum(
        np.abs(diff_model),
        dtype=np.float64
    )

    # Persistence errors
    diff_persistence = (
        y_persistence - y_true
    )

    persistence_se += np.sum(
        diff_persistence ** 2,
        dtype=np.float64
    )

    persistence_ae += np.sum(
        np.abs(diff_persistence),
        dtype=np.float64
    )

    n_pixels += y_true.size

    if (batch_number + 1) % 500 == 0:
        print(
            f"{batch_number + 1} / "
            f"{len(test_gen)} batches"
        )


model_mse = model_se / n_pixels
model_rmse = np.sqrt(model_mse)
model_mae = model_ae / n_pixels

persistence_mse = (
    persistence_se / n_pixels
)

persistence_rmse = np.sqrt(
    persistence_mse
)

persistence_mae = (
    persistence_ae / n_pixels
)

print("\nConvLSTM")
print("----------------")
print("MSE :", model_mse)
print("RMSE:", model_rmse)
print("MAE :", model_mae)

print("\nPersistence")
print("----------------")
print("MSE :", persistence_mse)
print("RMSE:", persistence_rmse)
print("MAE :", persistence_mae)





