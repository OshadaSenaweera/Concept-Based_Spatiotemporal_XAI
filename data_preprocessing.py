from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Extract data files should be put into folder named "data" in working directory

#Load the MeteoNet coordinate grid
coords = np.load("data/coords/radar_coords_NW.npz")
lats = coords["lats"]
lons = coords["lons"]

# Define the crop
lon_min, lon_max = -2, 1.9930000000000003
lat_min, lat_max = 46.251, 50.25

crop_mask = (
    (lons >= lon_min) &
    (lons <= lon_max) &
    (lats >= lat_min) &
    (lats <= lat_max)
)

rows, cols = np.where(crop_mask)

row_min = rows.min()
row_max = rows.max() + 1

col_min = cols.min()
col_max = cols.max() + 1

crop_height = row_max - row_min
crop_width = col_max - col_min

#Crop the coordinate arrays
lat_crop = lats[
    row_min:row_max,
    col_min:col_max
]

lon_crop = lons[
    row_min:row_max,
    col_min:col_max
]

#aggregating geographic coordinates

lat_200 = (
    lat_crop[0::2, 0::2] +
    lat_crop[1::2, 0::2] +
    lat_crop[0::2, 1::2] +
    lat_crop[1::2, 1::2]
) / 4.0


lon_200 = (
    lon_crop[0::2, 0::2] +
    lon_crop[1::2, 0::2] +
    lon_crop[0::2, 1::2] +
    lon_crop[1::2, 1::2]
) / 4.0

# Function for filling -1 pixels with neighbour averages, uses the valid eight surrounding spatial neighbours within the same radar frame.
def fill_missing_neighbor_mean(data):
    """
    Replace MeteoNet missing values (-1) with the mean
    of valid 8-neighbour spatial cells.

    Parameters
    ----------
    data : ndarray
        Shape (time, height, width)

    Returns
    -------
    ndarray
        float32 array with missing values filled.
    """

    data = data.astype(np.float32, copy=True)

    missing = data == -1

    if not np.any(missing):
        return data

    # Work with NaN internally
    x = data.copy()
    x[missing] = np.nan

    n, h, w = x.shape

    # Pad spatial dimensions only
    padded = np.pad(
        x,
        ((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan
    )

    neighbour_sum = np.zeros_like(x, dtype=np.float32)
    neighbour_count = np.zeros_like(x, dtype=np.float32)

    # Eight-neighbourhood
    for di in [-1, 0, 1]:

        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            neighbour = padded[
                :,
                1 + di : 1 + di + h,
                1 + dj : 1 + dj + w
            ]

            valid = ~np.isnan(neighbour)

            neighbour_sum += np.where(
                valid,
                neighbour,
                0.0
            )

            neighbour_count += valid.astype(np.float32)

    neighbour_mean = np.divide(
        neighbour_sum,
        neighbour_count,
        out=np.zeros_like(neighbour_sum),
        where=neighbour_count > 0
    )

    # Replace only the originally missing pixels
    data[missing] = neighbour_mean[missing]

    return data

# Function for 2 × 2 spatial averaging

def aggregate_2x2(data):
    """
    Aggregate 400x400 radar frames to 200x200 by
    averaging every non-overlapping 2x2 block.

    Input:
        (time, 400, 400)

    Output:
        (time, 200, 200)
    """

    if data.shape[1] != 400 or data.shape[2] != 400:
        raise ValueError(
            f"Expected spatial size 400x400, "
            f"got {data.shape[1:]}"
        )

    data_200 = (
        data[:, 0::2, 0::2] +
        data[:, 1::2, 0::2] +
        data[:, 0::2, 1::2] +
        data[:, 1::2, 1::2]
    ) / 4.0

    return data_200.astype(np.float32)

# function to load data in a memory-conscious manner
def load_and_process_radar(
    folder,
    row_min,
    row_max,
    col_min,
    col_max
):

    folder = Path(folder)

    # Only rainfall files
    files = sorted(
        folder.glob("rainfall_NW_*.npz")
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No rainfall_NW_*.npz files found in {folder}"
        )

    print("Number of radar files:", len(files))

    # --------------------------------------------------
    # PASS 1:
    # Determine total number of radar frames
    # --------------------------------------------------

    frame_counts = []
    file_date_ranges = []

    total_frames = 0

    for file in files:

        with np.load(file, allow_pickle=True) as radar:

            dates = pd.to_datetime(
                radar["dates"]
            )

            n_frames = len(dates)

        frame_counts.append(n_frames)

        file_date_ranges.append(
            (
                file.name,
                dates.min(),
                dates.max()
            )
        )

        total_frames += n_frames

    print("Total radar frames:", total_frames)

    # --------------------------------------------------
    # Allocate final output once
    # --------------------------------------------------

    radar_data = np.empty(
        (total_frames, 200, 200),
        dtype=np.float32
    )

    all_dates = np.empty(
        total_frames,
        dtype="datetime64[ns]"
    )

    # --------------------------------------------------
    # PASS 2:
    # Load -> crop -> fill -> aggregate -> store
    # --------------------------------------------------

    start = 0

    for file, n_frames in zip(files, frame_counts):

        print("\nProcessing:", file.name)

        with np.load(file, allow_pickle=True) as radar:

            # Crop before loading unnecessary spatial area
            values = radar["data"][
                :,
                row_min:row_max,
                col_min:col_max
            ]

            dates = pd.to_datetime(
                radar["dates"]
            )

        if len(values) != len(dates):
            raise ValueError(
                f"Data/date mismatch in {file.name}: "
                f"{len(values)} frames and "
                f"{len(dates)} dates"
            )

        print(
            "  Cropped:",
            values.shape
        )

        # Count missing pixels before filling
        n_missing = np.sum(values == -1)

        print(
            "  Missing pixels:",
            n_missing
        )

        # Fill missing radar pixels
        values = fill_missing_neighbor_mean(
            values
        )

        # Check
        remaining_missing = np.sum(
            values == -1
        )

        print(
            "  Remaining -1 values:",
            remaining_missing
        )

        # 400x400 -> 200x200
        values_200 = aggregate_2x2(
            values
        )

        print(
            "  Aggregated:",
            values_200.shape
        )

        end = start + n_frames

        radar_data[
            start:end
        ] = values_200

        all_dates[
            start:end
        ] = dates.to_numpy()

        print(
            "  Dates:",
            dates.min(),
            "to",
            dates.max()
        )

        start = end

        # Explicitly release large temporary arrays
        del values
        del values_200

    radar_dates = pd.DatetimeIndex(
        all_dates
    )

    return radar_data, radar_dates

#Load the processed dataset
radar_data, radar_dates = load_and_process_radar(
    folder="data",
    row_min=row_min,
    row_max=row_max,
    col_min=col_min,
    col_max=col_max
)

# check the loaded datset
print("\nFinal dataset")
print("------------------------")

print("Radar shape:", radar_data.shape)
print("dtype:", radar_data.dtype)

print(
    "Period:",
    radar_dates.min(),
    "to",
    radar_dates.max()
)

print(
    "Minimum radar value:",
    radar_data.min()
)

print(
    "Maximum radar value:",
    radar_data.max()
)

# Plot one processed radar frame
sample = 1000

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
    label="Radar value"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    f"MeteoNet radar\n"
    f"{radar_dates[sample]}"
)

plt.show()

#convert data into mm (milimeters)
radar_data /= np.float32(100.0)

# get average daily rainfall
frame_mean = radar_data.mean(axis=(1, 2), dtype=np.float64)

days = radar_dates.normalize()

daily_df = pd.DataFrame({
    "date": days,
    "frame_mean": frame_mean
})

daily_summary = (
    daily_df
    .groupby("date")
    .agg(
        rainfall_score=("frame_mean", "mean"),
        n_scans=("frame_mean", "size")
    )
    .reset_index()
)

# get top 100 tainy days radar frames only
top_100_days = (
    daily_summary
    .sort_values("rainfall_score", ascending=False)
    .head(100)
    .reset_index(drop=True)
)
selected_days = top_100_days["date"]
mask = days.isin(selected_days)
selected_idx = np.flatnonzero(mask)

import gc

selected_shape = (
    len(selected_idx),
    radar_data.shape[1],
    radar_data.shape[2]
)

print("Selected shape:", selected_shape)

radar_selected = np.memmap(
    "radar_selected.dat",
    dtype=np.float32,
    mode="w+",
    shape=selected_shape
)

chunk_size = 500

for start in range(0, len(selected_idx), chunk_size):
    end = min(start + chunk_size, len(selected_idx))

    idx_chunk = selected_idx[start:end]

    radar_selected[start:end] = radar_data[idx_chunk]

    print(f"Copied {end} / {len(selected_idx)} frames")

radar_selected.flush()

### Save Data
dates_selected = radar_dates[selected_idx]

np.save(
    "radar_selected_dates.npy",
    dates_selected.to_numpy()
)

radar_selected = np.memmap(
    "radar_selected.dat",
    dtype=np.float32,
    mode="r",
    shape=(28800, 200, 200)
)

np.save(
    "radar_selected.npy",
    radar_selected
) 

np.save(
    "radar_dates_selected.npy",
    np.asarray(dates_selected, dtype="datetime64[ns]")
)
np.save(
    "lat_200.npy",
    lat_200.astype(np.float32)
)

np.save(
    "lon_200.npy",
    lon_200.astype(np.float32)
)

top_100_days.to_csv(
    "top_100_rainy_days.csv",
    index=False
)




















