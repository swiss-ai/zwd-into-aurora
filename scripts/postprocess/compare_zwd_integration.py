"""
Compare ZWD from check_zarr integration against an external reference ZWD.

Usage:
    python scripts/postprocess/compare_zwd_integration.py \
        --target /path/to/target.zarr \
        --external /path/to/zwd_20200801.zarr \
        --static static_vars.nc \
        --date 2020-08-01
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Reuse the integration from check_zarr
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from check_zarr_for_zwd import compute_zwd_integrated


def main():
    parser = argparse.ArgumentParser(description="Compare ZWD integrations")
    parser.add_argument("--target", required=True, help="Aurora target zarr (has q, T, mslp)")
    parser.add_argument("--external", required=True, help="External ZWD zarr (has 'zwd' variable)")
    parser.add_argument("--static", default=None, help="Static file with geopotential_at_surface")
    parser.add_argument("--date", default="2020-08-01", help="Date to compare (YYYY-MM-DD)")
    parser.add_argument("--plot-dir", default="compare_zwd_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    # --- Load external reference ZWD ---
    print(f"Loading external ZWD: {args.external}")
    ds_ext = xr.open_zarr(args.external)
    zwd_ext = ds_ext["zwd"]
    print(f"  Shape: {zwd_ext.shape}, dims: {zwd_ext.dims}")
    print(f"  Range: [{float(zwd_ext.min()):.1f}, {float(zwd_ext.max()):.1f}] mm")
    print(f"  Lat: {float(ds_ext.latitude[0]):.2f} to {float(ds_ext.latitude[-1]):.2f} ({len(ds_ext.latitude)} pts)")

    # --- Load static geopotential ---
    z_sfc = None
    if args.static:
        print(f"\nLoading static: {args.static}")
        if args.static.endswith(".nc"):
            static_ds = xr.open_dataset(args.static)
        else:
            static_ds = xr.open_zarr(args.static)
        for vname in ["geopotential_at_surface", "z", "orography"]:
            if vname in static_ds:
                z_sfc = static_ds[vname]
                for dim in ["time", "init_time"]:
                    if dim in z_sfc.dims:
                        z_sfc = z_sfc.isel({dim: 0})
                z_sfc = z_sfc.compute()
                print(f"  Loaded '{vname}'")
                break

    # --- Load target and select matching date ---
    print(f"\nLoading target: {args.target}")
    ds_tgt = xr.open_zarr(args.target)
    print(f"  Total init_times: {ds_tgt.sizes['init_time']}")

    # Select times matching the requested date
    init_times = ds_tgt.init_time.values
    date_start = np.datetime64(args.date)
    date_end = date_start + np.timedelta64(1, "D")
    mask = (init_times >= date_start) & (init_times < date_end)
    matching_idx = np.where(mask)[0]

    if len(matching_idx) == 0:
        print(f"ERROR: No init_times found for {args.date}")
        print(f"  Available range: {init_times[0]} to {init_times[-1]}")
        return

    ds_day = ds_tgt.isel(init_time=matching_idx)
    print(f"  Selected {len(matching_idx)} times for {args.date}")

    # --- Compute ZWD via check_zarr integration ---
    print("\nComputing ZWD integration (check_zarr method)...")
    zwd_integrated = compute_zwd_integrated(ds_day, label="check_zarr", z_sfc=z_sfc)

    # --- Align grids (external may have 721 lats vs 720) ---
    zwd_ext_day = zwd_ext.compute()

    # Match times: external uses init_time too
    ext_times = zwd_ext_day.init_time.values
    int_times = zwd_integrated.init_time.values
    common_times = np.intersect1d(ext_times, int_times)
    if len(common_times) == 0:
        # Try matching by hour of day
        print("  No exact time match; aligning by count (assuming same temporal order)")
        n_common = min(len(ext_times), len(int_times))
        zwd_ext_sel = zwd_ext_day.isel(init_time=slice(0, n_common))
        zwd_int_sel = zwd_integrated.isel(init_time=slice(0, n_common))
    else:
        print(f"  {len(common_times)} matching times found")
        zwd_ext_sel = zwd_ext_day.sel(init_time=common_times)
        zwd_int_sel = zwd_integrated.sel(init_time=common_times)

    # Interpolate external to target lat/lon grid if different
    tgt_lat = zwd_int_sel.latitude.values
    tgt_lon = zwd_int_sel.longitude.values
    ext_lat = zwd_ext_sel.latitude.values

    if len(ext_lat) != len(tgt_lat) or not np.allclose(ext_lat, tgt_lat, atol=0.01):
        print(f"  Regridding external ({len(ext_lat)} lats) -> target ({len(tgt_lat)} lats)")
        zwd_ext_sel = zwd_ext_sel.interp(latitude=tgt_lat, longitude=tgt_lon, method="linear")

    # Squeeze lead_time if present
    if "lead_time" in zwd_ext_sel.dims:
        zwd_ext_sel = zwd_ext_sel.squeeze("lead_time", drop=True)
    if "lead_time" in zwd_int_sel.dims:
        zwd_int_sel = zwd_int_sel.squeeze("lead_time", drop=True)

    # --- Also get the target's own zenith_wet_delay if available ---
    has_tgt_zwd = "zenith_wet_delay" in ds_day
    if has_tgt_zwd:
        zwd_tgt_direct = ds_day["zenith_wet_delay"].compute()
        if len(common_times) > 0:
            zwd_tgt_direct = zwd_tgt_direct.sel(init_time=common_times)
        else:
            zwd_tgt_direct = zwd_tgt_direct.isel(init_time=slice(0, n_common))
        if "lead_time" in zwd_tgt_direct.dims:
            zwd_tgt_direct = zwd_tgt_direct.squeeze("lead_time", drop=True)

    # --- Statistics ---
    diff = zwd_int_sel - zwd_ext_sel
    print(f"\n{'='*60}")
    print("COMPARISON: check_zarr integration vs external reference")
    print(f"{'='*60}")
    print(f"  External ZWD:    mean={float(zwd_ext_sel.mean()):.2f} mm")
    print(f"  Integrated ZWD:  mean={float(zwd_int_sel.mean()):.2f} mm")
    print(f"  Bias (int-ext):  {float(diff.mean()):+.2f} mm")
    print(f"  RMSE:            {float(np.sqrt((diff**2).mean())):.2f} mm")
    print(f"  Rel. bias:       {float(diff.mean()/zwd_ext_sel.mean())*100:+.2f}%")

    if has_tgt_zwd:
        diff_tgt = zwd_tgt_direct - zwd_ext_sel
        print(f"\n  Target ZWD var:  mean={float(zwd_tgt_direct.mean()):.2f} mm")
        print(f"  Bias (tgt-ext):  {float(diff_tgt.mean()):+.2f} mm")
        print(f"  RMSE:            {float(np.sqrt((diff_tgt**2).mean())):.2f} mm")

    # --- Plot ---
    time_dims = [d for d in diff.dims if d in ("init_time",)]
    mean_ext = zwd_ext_sel.mean(dim=time_dims)
    mean_int = zwd_int_sel.mean(dim=time_dims)
    mean_diff = diff.mean(dim=time_dims)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.suptitle(f"ZWD Integration Comparison — {args.date}", fontsize=14)

    def plot_panel(ax, data, title, cmap, vmin=None, vmax=None, symmetric=False):
        if symmetric and vmax is not None:
            vmin = -vmax
        lats = data.latitude.values
        lons = data.longitude.values
        plot_data = data
        if symmetric and len(lats) > 360:
            plot_data = data.coarsen(latitude=2, longitude=2, boundary="trim").mean()
            lats = plot_data.latitude.values
            lons = plot_data.longitude.values
        im = ax.pcolormesh(lons, lats, plot_data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
        ax.coastlines(linewidth=0.5)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    plot_panel(axes[0, 0], mean_ext, "External reference [mm]", "Blues",
               vmin=0, vmax=float(mean_ext.quantile(0.98)))
    plot_panel(axes[0, 1], mean_int, "check_zarr integration [mm]", "Blues",
               vmin=0, vmax=float(mean_ext.quantile(0.98)))

    vmax_bias = max(abs(float(mean_diff.quantile(0.02))),
                    abs(float(mean_diff.quantile(0.98))), 0.1)
    plot_panel(axes[1, 0], mean_diff, "Bias (integrated - external) [mm]",
               "RdBu_r", vmax=vmax_bias, symmetric=True)

    # Scatter
    axes[1, 1].remove()
    ax_sc = fig.add_subplot(2, 2, 4)
    ext_flat = mean_ext.values.ravel()
    int_flat = mean_int.values.ravel()
    valid = np.isfinite(ext_flat) & np.isfinite(int_flat)
    ext_v, int_v = ext_flat[valid], int_flat[valid]
    if len(ext_v) > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ext_v), 200_000, replace=False)
        ext_v, int_v = ext_v[idx], int_v[idx]
    ax_sc.scatter(ext_v, int_v, s=0.3, alpha=0.1)
    lims = [min(ext_v.min(), int_v.min()), max(ext_v.max(), int_v.max())]
    ax_sc.plot(lims, lims, "r--", lw=1, label="1:1")
    ax_sc.set_xlabel("External ZWD [mm]")
    ax_sc.set_ylabel("Integrated ZWD [mm]")
    ax_sc.set_title("Scatter (time-mean per grid cell)")
    ax_sc.legend()

    plt.tight_layout()
    fname = os.path.join(args.plot_dir, f"zwd_integration_comparison_{args.date}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
