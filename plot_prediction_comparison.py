import matplotlib
matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# -----------------------
# CONFIGURATION
# -----------------------
workspace = "/path/to/outputs/"
ground_truth_path = os.path.join(workspace, "target_2020-10-10_5steps.zarr")
prediction_path = os.path.join(workspace, "pred_2020-10-10_5steps.zarr")
time_index = 0  # Change this to select a different timestep
output_dir = os.path.join(workspace, "comparison_plots")
colormap = "viridis"

# -----------------------
# LOAD DATA
# -----------------------
ds_gt = xr.open_zarr(ground_truth_path)
ds_pred = xr.open_zarr(prediction_path)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get time string
time_str = str(ds_gt['time'].isel(time=time_index).values)[:16].replace(":", "-")

# -----------------------
# PLOTTING FUNCTION
# -----------------------
# def plot_and_save(gt, pred, diff, varname):
#     fig = plt.figure(figsize=(18, 10))

#     # Set consistent vmin/vmax for fair comparison
#     vmin = min(gt.min().compute().values, pred.min().compute().values)
#     vmax = max(gt.max().compute().values, pred.max().compute().values)


#     def plot_map(data, title, vmin=None, vmax=None, cmap=colormap):
#         ax = plt.axes(projection=ccrs.PlateCarree())
#         data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
#                   cbar_kwargs={'shrink': 0.5}, vmin=vmin, vmax=vmax)
#         ax.coastlines()
#         ax.add_feature(cfeature.BORDERS, linewidth=0.5)
#         ax.set_title(title)

#     plt.subplot(1, 3, 1)
#     plot_map(gt, f"Ground Truth\n{varname}\n{time_str}", vmin=vmin, vmax=vmax)

#     plt.subplot(1, 3, 2)
#     plot_map(pred, f"Prediction\n{varname}\n{time_str}", vmin=vmin, vmax=vmax)

#     plt.subplot(1, 3, 3)
#     plot_map(diff, f"Prediction - Ground Truth\n{varname}\n{time_str}", cmap="coolwarm")

#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{varname}_{time_str}.png")
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"Saved: {out_path}")

def plot_and_save(gt, pred, err, varname):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Set consistent vmin/vmax for fair comparison
    vmin = min(gt.min().compute().values, pred.min().compute().values)
    vmax = max(gt.max().compute().values, pred.max().compute().values)

    def plot_map(ax, data, title, vmin=None, vmax=None, cmap=colormap):
        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                  cbar_kwargs={'shrink': 0.5}, vmin=vmin, vmax=vmax, rasterized=True) # rasterized=True for large datasets
        ax.coastlines(resolution='110m')  # low resolution, instead of '10m'

        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_title(title)

    plot_map(axes[0], gt, f"Ground Truth\n{varname}\n{time_str}", vmin=vmin, vmax=vmax)
    plot_map(axes[1], pred, f"Prediction\n{varname}\n{time_str}", vmin=vmin, vmax=vmax)
    plot_map(axes[2], err, f"Absolute error\n{varname}\n{time_str}", cmap="coolwarm")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{varname}_{time_str}.svg")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------
# LOOP THROUGH VARIABLES
# -----------------------
for var in ds_gt.data_vars:
    if var not in ds_pred:
        print(f"Skipping '{var}': not found in predictions.")
        continue

    gt = ds_gt[var].isel(time=time_index)
    pred = ds_pred[var].isel(time=time_index)

    eps = 1e-6  # Small value to avoid division by zero
    err = abs(gt - pred)

    plot_and_save(gt, pred, err, var)
