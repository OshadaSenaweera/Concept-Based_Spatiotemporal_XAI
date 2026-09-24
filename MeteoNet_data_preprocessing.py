from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


























