from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# Extracted MeteoNet rainfall files should be placed in:
# data/
#
# The coordinate file should be placed in:
# data/coords/radar_coords_NW.npz

DATA_DIR = Path("data")
COORDS_FILE = DATA_DIR / "coords" / "radar_coords_NW.npz"

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geographic bounds used to obtain the 400 x 400 radar crop
LON_MIN, LON_MAX = -2.0, 1.993
LAT_MIN, LAT_MAX = 46.251, 50.25

# Number of rainy days to retain
N_SELECTED_DAYS = 100

# Expected number of 5-minute radar scans in a complete day
SCANS_PER_DAY = 288


# ============================================================
# Load MeteoNet coordinate grid
# ============================================================

coords = np.load(COORDS_FILE)

lats = coords["lats"]
lons = coords["lons"]

print("Original coordinate grid:", lats.shape)


# ============================================================
# Determine spatial crop
# ============================================================

crop_mask = (
    (lons >= LON_MIN)
    & (lons <= LON_MAX)
    & (lats >= LAT_MIN)
    & (lats <= LAT_MAX)
)

rows, cols = np.where(crop_mask)

if len(rows) == 0 or len(cols) == 0:
    raise ValueError(
        "The specified geographic bounds do not intersect "
        "the MeteoNet coordinate grid."
    )

row_min = rows.min()
row_max = rows.max() + 1

col_min = cols.min()
col_max = cols.max() + 1

crop_height = row_max - row_min
crop_width = col_max - col_min

print(
    f"Crop size: {crop_height} x {crop_width}"
)

# The subsequent aggregation assumes a 400 x 400 crop
if crop_height != 400 or crop_width != 400:
    raise ValueError(
        f"Expected a 400x400 crop, but obtained "
        f"{crop_height}x{crop_width}."
    )


# ============================================================
# Crop coordinate arrays
# ============================================================

lat_crop = lats[
    row_min:row_max,
    col_min:col_max
]

lon_crop = lons[
    row_min:row_max,
    col_min:col_max
]


# ============================================================
# Aggregate coordinates from 400 x 400 to 200 x 200
# ============================================================

lat_200 = (
    lat_crop[0::2, 0::2]
    + lat_crop[1::2, 0::2]
    + lat_crop[0::2, 1::2]
    + lat_crop[1::2, 1::2]
) / 4.0

lon_200 = (
    lon_crop[0::2, 0::2]
    + lon_crop[1::2, 0::2]
    + lon_crop[0::2, 1::2]
    + lon_crop[1::2, 1::2]
) / 4.0

print(
    "Aggregated coordinate grid:",
    lat_200.shape
)


# ============================================================
# Missing-value filling
# ============================================================

def fill_missing_neighbor_mean(data):
    """
    Replace MeteoNet missing values (-1) using the mean of
    valid cells in the 8-neighbour spatial neighbourhood
    within the same radar frame.

    If a missing cell has no valid spatial neighbours,
    it is filled with 0.

    Parameters
    ----------
    data : numpy.ndarray
        Radar data with shape (time, height, width).

    Returns
    -------
    numpy.ndarray
        Float32 array with missing values filled.
    """

    data = data.astype(
        np.float32,
        copy=True
    )

    missing = data == -1

    if not np.any(missing):
        return data

    # Use NaN internally to identify invalid neighbours
    x = data.copy()
    x[missing] = np.nan

    _, h, w = x.shape

    # Pad only the spatial dimensions
    padded = np.pad(
        x,
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan
    )

    neighbour_sum = np.zeros_like(
        x,
        dtype=np.float32
    )

    neighbour_count = np.zeros_like(
        x,
        dtype=np.float32
    )

    # Eight-neighbour spatial neighbourhood
    for di in [-1, 0, 1]:

        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            neighbour = padded[
                :,
                1 + di:1 + di + h,
                1 + dj:1 + dj + w
            ]

            valid = ~np.isnan(neighbour)

            neighbour_sum += np.where(
                valid,
                neighbour,
                0.0
            )

            neighbour_count += (
                valid.astype(np.float32)
            )

    # Cells with no valid neighbours receive 0
    neighbour_mean = np.divide(
        neighbour_sum,
        neighbour_count,
        out=np.zeros_like(neighbour_sum),
        where=neighbour_count > 0
    )

    # Replace only originally missing cells
    data[missing] = neighbour_mean[missing]

    return data


# ============================================================
# 2 x 2 spatial aggregation
# ============================================================

def aggregate_2x2(data):
    """
    Aggregate 400 x 400 radar frames to 200 x 200 by
    averaging every non-overlapping 2 x 2 spatial block.

    Parameters
    ----------
    data : numpy.ndarray
        Radar data with shape (time, 400, 400).

    Returns
    -------
    numpy.ndarray
        Aggregated float32 array with shape
        (time, 200, 200).
    """

    if data.shape[1:] != (400, 400):
        raise ValueError(
            f"Expected spatial size 400x400, "
            f"but obtained {data.shape[1:]}."
        )

    data_200 = (
        data[:, 0::2, 0::2]
        + data[:, 1::2, 0::2]
        + data[:, 0::2, 1::2]
        + data[:, 1::2, 1::2]
    ) / 4.0

    return data_200.astype(
        np.float32
    )


# ============================================================
# Load and process radar files
# ============================================================

def load_and_process_radar(
    folder,
    row_min,
    row_max,
    col_min,
    col_max
):
    """
    Load MeteoNet radar files and perform spatial
    preprocessing.

    Processing steps:
        1. Extract 400 x 400 spatial crop.
        2. Fill missing (-1) radar pixels.
        3. Aggregate to 200 x 200.
        4. Combine all files chronologically.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing rainfall_NW_*.npz files.

    row_min, row_max, col_min, col_max : int
        Spatial crop indices.

    Returns
    -------
    radar_data : numpy.ndarray
        Processed radar array with shape
        (time, 200, 200).

    radar_dates : pandas.DatetimeIndex
        Corresponding radar timestamps.
    """

    folder = Path(folder)

    files = sorted(
        folder.glob("rainfall_NW_*.npz")
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No rainfall_NW_*.npz files "
            f"found in {folder}."
        )

    print(
        "\nNumber of radar files:",
        len(files)
    )

    # --------------------------------------------------------
    # PASS 1:
    # Determine total number of radar frames
    # --------------------------------------------------------

    frame_counts = []
    total_frames = 0

    for file in files:

        with np.load(
            file,
            allow_pickle=True
        ) as radar:

            dates = pd.to_datetime(
                radar["dates"]
            )

            n_frames = len(dates)

        frame_counts.append(n_frames)
        total_frames += n_frames

    print(
        "Total radar frames:",
        total_frames
    )

    # --------------------------------------------------------
    # Allocate output arrays once
    # --------------------------------------------------------

    radar_data = np.empty(
        (total_frames, 200, 200),
        dtype=np.float32
    )

    all_dates = np.empty(
        total_frames,
        dtype="datetime64[ns]"
    )

    # --------------------------------------------------------
    # PASS 2:
    # Load -> crop -> fill -> aggregate -> store
    # --------------------------------------------------------

    start = 0

    for file, n_frames in zip(
        files,
        frame_counts
    ):

        print(
            "\nProcessing:",
            file.name
        )

        with np.load(
            file,
            allow_pickle=True
        ) as radar:

            # Extract the required spatial crop
            values = radar["data"][
                :,
                row_min:row_max,
                col_min:col_max
            ]

            dates = pd.to_datetime(
                radar["dates"]
            )

        # Verify matching data and timestamps
        if len(values) != len(dates):
            raise ValueError(
                f"Data/date mismatch in {file.name}: "
                f"{len(values)} frames and "
                f"{len(dates)} dates."
            )

        print(
            "  Cropped:",
            values.shape
        )

        # Count missing pixels before filling
        n_missing = np.count_nonzero(
            values == -1
        )

        print(
            "  Missing pixels:",
            n_missing
        )

        # Fill missing radar pixels
        values = fill_missing_neighbor_mean(
            values
        )

        # Verify that -1 values have been removed
        remaining_missing = np.count_nonzero(
            values == -1
        )

        print(
            "  Remaining -1 values:",
            remaining_missing
        )

        # Aggregate from 400 x 400 to 200 x 200
        values_200 = aggregate_2x2(
            values
        )

        print(
            "  Aggregated:",
            values_200.shape
        )

        # Store results
        end = start + n_frames

        radar_data[
            start:end
        ] = values_200

        all_dates[
            start:end
        ] = dates.to_numpy(
            dtype="datetime64[ns]"
        )

        print(
            "  Dates:",
            dates.min(),
            "to",
            dates.max()
        )

        start = end

        # Release large temporary arrays
        del values
        del values_200

    # Final consistency check
    if start != total_frames:
        raise RuntimeError(
            f"Expected {total_frames} frames, "
            f"but stored {start}."
        )

    radar_dates = pd.DatetimeIndex(
        all_dates
    )

    return radar_data, radar_dates


# ============================================================
# Process the complete radar dataset
# ============================================================

radar_data, radar_dates = load_and_process_radar(
    folder=DATA_DIR,
    row_min=row_min,
    row_max=row_max,
    col_min=col_min,
    col_max=col_max
)


# ============================================================
# Check processed dataset
# ============================================================

print("\nFinal dataset")
print("------------------------")

print(
    "Radar shape:",
    radar_data.shape
)

print(
    "Data type:",
    radar_data.dtype
)

print(
    "Period:",
    radar_dates.min(),
    "to",
    radar_dates.max()
)

print(
    "Minimum raw radar value:",
    radar_data.min()
)

print(
    "Maximum raw radar value:",
    radar_data.max()
)


# ============================================================
# Convert radar values to millimetres
# ============================================================

radar_data /= np.float32(100.0)

print(
    "Minimum rainfall (mm):",
    radar_data.min()
)

print(
    "Maximum rainfall (mm):",
    radar_data.max()
)


# ============================================================
# Plot one processed radar frame
# ============================================================

sample = 1000

if sample >= len(radar_data):
    raise IndexError(
        f"Sample index {sample} exceeds the "
        f"dataset size of {len(radar_data)}."
    )

fig, ax = plt.subplots(
    figsize=(8, 7)
)

mesh = ax.pcolormesh(
    lon_200,
    lat_200,
    radar_data[sample],
    shading="auto"
)

plt.colorbar(
    mesh,
    ax=ax,
    label="Rainfall (mm)"
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)

ax.set_title(
    f"MeteoNet radar\n"
    f"{radar_dates[sample]}"
)

plt.tight_layout()
plt.show()


# ============================================================
# Calculate daily rainfall scores
# ============================================================

# Calculate the spatial mean rainfall for each radar frame
frame_mean = radar_data.mean(
    axis=(1, 2),
    dtype=np.float64
)

# Assign each frame to its calendar day
days = radar_dates.normalize()

daily_df = pd.DataFrame({
    "date": days,
    "frame_mean": frame_mean
})

# The rainfall score is the mean spatial rainfall across
# all radar frames available for each day.
daily_summary = (
    daily_df
    .groupby("date")
    .agg(
        rainfall_score=(
            "frame_mean",
            "mean"
        ),
        n_scans=(
            "frame_mean",
            "size"
        )
    )
    .reset_index()
)

print(
    "\nNumber of days:",
    len(daily_summary)
)

print(
    "Complete days:",
    np.sum(
        daily_summary["n_scans"]
        == SCANS_PER_DAY
    )
)


# ============================================================
# Select the 100 rainiest complete days
# ============================================================

# A complete day contains 288 five-minute radar scans.
complete_days = daily_summary[
    daily_summary["n_scans"]
    == SCANS_PER_DAY
].copy()

if len(complete_days) < N_SELECTED_DAYS:
    raise ValueError(
        f"Only {len(complete_days)} complete days "
        f"are available, but {N_SELECTED_DAYS} "
        f"were requested."
    )

top_100_days = (
    complete_days
    .sort_values(
        "rainfall_score",
        ascending=False
    )
    .head(N_SELECTED_DAYS)
    .reset_index(drop=True)
)

selected_days = top_100_days["date"]

print(
    "\nSelected days:",
    len(selected_days)
)

print(
    "Expected selected frames:",
    N_SELECTED_DAYS * SCANS_PER_DAY
)


# ============================================================
# Extract frames belonging to selected days
# ============================================================

mask = days.isin(
    selected_days
)

# np.flatnonzero returns the selected frame indices in
# their original chronological order.
selected_idx = np.flatnonzero(
    mask
)

selected_shape = (
    len(selected_idx),
    radar_data.shape[1],
    radar_data.shape[2]
)

print(
    "Selected radar shape:",
    selected_shape
)

expected_frames = (
    N_SELECTED_DAYS
    * SCANS_PER_DAY
)

if len(selected_idx) != expected_frames:
    raise RuntimeError(
        f"Expected {expected_frames} selected frames, "
        f"but found {len(selected_idx)}."
    )


# ============================================================
# Copy selected radar data using a memory-mapped array
# ============================================================

memmap_path = (
    OUTPUT_DIR
    / "radar_selected_temp.dat"
)

radar_selected = np.memmap(
    memmap_path,
    dtype=np.float32,
    mode="w+",
    shape=selected_shape
)

chunk_size = 500

for start in range(
    0,
    len(selected_idx),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(selected_idx)
    )

    idx_chunk = selected_idx[
        start:end
    ]

    radar_selected[
        start:end
    ] = radar_data[
        idx_chunk
    ]

    print(
        f"Copied {end} / "
        f"{len(selected_idx)} frames"
    )

radar_selected.flush()


# ============================================================
# Save processed data
# ============================================================

dates_selected = radar_dates[
    selected_idx
]

# Save radar array
np.save(
    OUTPUT_DIR / "radar_selected.npy",
    radar_selected
)

# Save corresponding timestamps
np.save(
    OUTPUT_DIR / "radar_dates_selected.npy",
    dates_selected.to_numpy(
        dtype="datetime64[ns]"
    )
)

# Save aggregated geographic coordinates
np.save(
    OUTPUT_DIR / "lat_200.npy",
    lat_200.astype(np.float32)
)

np.save(
    OUTPUT_DIR / "lon_200.npy",
    lon_200.astype(np.float32)
)

# Save information about the selected rainy days
top_100_days.to_csv(
    OUTPUT_DIR / "top_100_rainy_days.csv",
    index=False
)


# ============================================================
# Remove temporary memory-mapped file
# ============================================================

del radar_selected

if memmap_path.exists():
    memmap_path.unlink()


# ============================================================
# Final summary
# ============================================================

print("\nPreprocessing complete")
print("------------------------")

print(
    "Selected radar frames:",
    len(selected_idx)
)

print(
    "Selected radar shape:",
    selected_shape
)

print(
    "Selected period:",
    dates_selected.min(),
    "to",
    dates_selected.max()
)

print(
    "Output directory:",
    OUTPUT_DIR.resolve()
)

print("\nSaved files:")
print("  radar_selected.npy")
print("  radar_dates_selected.npy")
print("  lat_200.npy")
print("  lon_200.npy")
print("  top_100_rainy_days.csv")
