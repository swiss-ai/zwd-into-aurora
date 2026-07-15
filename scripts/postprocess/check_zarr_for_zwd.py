"""
Compare ZWD from Aurora predictions against target ground truth.

Three ZWD sources are compared against the target's zenith_wet_delay:
  1. Prediction integrated  — ZWD from vertically integrating Finetunedq, T profiles
  2. Prediction predicted   — ZWD directly predicted by the model
  3. Baseline integrated    — ZWD from vertically integrating baseline q, T profiles

Usage:
    python scripts/postprocess/check_zarr_for_zwd.py \
        --Finetunedpred.zarr --target target.zarr [--baseline baseline.zarr]
"""

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# CONSTANTS
# ============================================================
G         = 9.80665        # standard gravity [m/s²]
R_D       = 287.05         # specific gas constant for dry air [J/(kg·K)]
R_V       = 461.5          # specific gas constant for water vapour [J/(kg·K)]
K1        = 77.689         # refractivity constant [K/hPa] (Rueger 2002)
K2        = 71.295         # refractivity constant [K/hPa] (Rueger 2002)
K3        = 375463.0       # refractivity constant [K²/hPa] (Rueger 2002)
K2_PRIME  = K2 - K1 * (18.0153 / 28.9644)  # ≈ 22.97, for IWV-based formula only
T_LAPSE   = 0.0065         # standard temperature lapse rate [K/m]

# Aurora pressure levels [hPa]
PRESSURE_LEVELS_HPA = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dtype=np.float64,
)
PRESSURE_LEVELS_PA = PRESSURE_LEVELS_HPA * 100.0


# ============================================================
# STRUCTURE CHECK
# ============================================================
def check_structure(ds: xr.Dataset, label: str = "") -> tuple:
    """
    Verify the zarr has the required variables and dimensions.

    Returns:
        (ok, has_zwd): ok=True if minimum required vars exist,
                       has_zwd=True if zenith_wet_delay is present.
    """
    ok = True
    prefix = f"[{label}] " if label else ""

    required_atmos = {"specific_humidity": ("init_time", "lead_time", "level", "latitude", "longitude"),
                      "temperature":       ("init_time", "lead_time", "level", "latitude", "longitude")}
    required_surf  = {"mean_sea_level_pressure": ("init_time", "lead_time", "latitude", "longitude")}
    optional_surf  = {"zenith_wet_delay":        ("init_time", "lead_time", "latitude", "longitude")}

    for var, expected_dims in {**required_atmos, **required_surf}.items():
        if var not in ds.data_vars:
            print(f"  {prefix}FAIL: Missing required variable '{var}'")
            ok = False
            continue
        actual = tuple(ds[var].dims)
        if actual != expected_dims:
            print(f"  {prefix}WARN: '{var}' dims {actual}, expected {expected_dims}")
        else:
            print(f"  {prefix}OK  : '{var}' {actual} shape={ds[var].shape}")

    has_zwd = False
    for var, expected_dims in optional_surf.items():
        if var not in ds.data_vars:
            print(f"  {prefix}INFO: '{var}' not present")
        else:
            has_zwd = True
            actual = tuple(ds[var].dims)
            if actual != expected_dims:
                print(f"  {prefix}WARN: '{var}' dims {actual}, expected {expected_dims}")
            else:
                print(f"  {prefix}OK  : '{var}' {actual} shape={ds[var].shape}")

    if "level" in ds.coords:
        levels = ds.level.values.astype(float)
        if levels.max() > 2000:
            print(f"  {prefix}INFO: Levels appear to be in Pa, will convert to hPa internally.")
        elif np.allclose(np.sort(levels), np.sort(PRESSURE_LEVELS_HPA)):
            print(f"  {prefix}OK  : Pressure levels match Aurora 13-level config (hPa).")
        else:
            print(f"  {prefix}WARN: Unexpected pressure levels: {levels}")

    if "specific_humidity" in ds:
        q_sample = ds["specific_humidity"].isel(init_time=0, lead_time=0)
        qmin, qmax = float(q_sample.min()), float(q_sample.max())
        if qmin < -1e-6:
            print(f"  {prefix}WARN: specific_humidity has negative values (min={qmin:.2e})")
        print(f"  {prefix}INFO: q range [{qmin:.2e}, {qmax:.2e}] kg/kg")

    if has_zwd:
        zwd_sample = ds["zenith_wet_delay"].isel(init_time=0, lead_time=0)
        zwdmin, zwdmax = float(zwd_sample.min()), float(zwd_sample.max())
        print(f"  {prefix}INFO: zenith_wet_delay range [{zwdmin:.1f}, {zwdmax:.1f}] mm")
        if zwdmax < 1.0:
            print(f"  {prefix}WARN: ZWD values < 1 — might be in meters, not mm?")

    return ok, has_zwd


# ============================================================
# PHYSICS COMPUTATIONS
# ============================================================
def estimate_surface_pressure(mslp: xr.DataArray, z_sfc: xr.DataArray,
                              t_sfc: xr.DataArray = None) -> xr.DataArray:
    """
    Estimate actual surface pressure from MSLP and surface geopotential.

    Uses the barometric formula (hypsometric equation):
        Ps = MSLP * (1 - L*h / T0) ^ (g / (R_d * L))

    where h = z_sfc / g is the surface elevation [m],
    L = 0.0065 K/m (standard lapse rate), T0 = MSL temperature (~288.15 K).

    If t_sfc (2m temperature) is provided, it is used as a better estimate
    of the mean column temperature instead of the standard 288.15 K.

    Args:
        mslp: Mean sea level pressure [Pa].
        z_sfc: Surface geopotential [m²/s²].
        t_sfc: Optional 2m temperature [K] for better accuracy.

    Returns:
        Estimated surface pressure [Pa].
    """
    h = z_sfc / G  # surface elevation [m]
    T0 = t_sfc if t_sfc is not None else 288.15
    exponent = G / (R_D * T_LAPSE)
    ps = mslp * (1.0 - T_LAPSE * h / T0) ** exponent
    return ps


def compute_dp(pressure_pa: np.ndarray, surface_pressure: xr.DataArray,
               level_dim: str = "level") -> xr.DataArray:
    """
    Trapezoidal pressure layer thicknesses [Pa].
    Levels below the surface pressure are masked to zero.
    The lowest above-surface level has its dp extended down to the surface pressure.
    """
    n = len(pressure_pa)
    dp_vals = np.zeros(n)
    dp_vals[0]  = 0.5 * (pressure_pa[1] - pressure_pa[0])
    dp_vals[-1] = 0.5 * (pressure_pa[-1] - pressure_pa[-2])
    for k in range(1, n - 1):
        dp_vals[k] = 0.5 * (pressure_pa[k + 1] - pressure_pa[k - 1])

    p_da = xr.DataArray(pressure_pa, dims=[level_dim],
                        coords={level_dim: pressure_pa / 100.0})  # coord in hPa
    dp_da = xr.DataArray(dp_vals, dims=[level_dim],
                         coords={level_dim: pressure_pa / 100.0})

    # Mask underground levels (p > ps)
    mask = p_da <= surface_pressure
    dp_masked = dp_da.where(mask, other=0.0)

    # Adjust the lowest above-surface level: extend its dp down to p_surface.
    # For each grid point, the lowest valid level k has its lower boundary
    # clamped to min(p[k+1], ps) instead of p[k+1].
    # This recovers the partial layer between ps and the next level above.
    for k in range(n - 1, -1, -1):
        is_lowest_above = (pressure_pa[k] <= surface_pressure)
        if k < n - 1:
            is_lowest_above = is_lowest_above & (pressure_pa[k + 1] > surface_pressure)
        # Recompute dp for this level: upper boundary unchanged, lower = ps
        if k == 0:
            upper_edge = pressure_pa[0] - 0.5 * (pressure_pa[1] - pressure_pa[0])
        else:
            upper_edge = 0.5 * (pressure_pa[k] + pressure_pa[k - 1])
        dp_adjusted = surface_pressure - upper_edge
        # Clamp to original dp (in case ps is between this level and next)
        dp_adjusted = dp_adjusted.clip(min=0.0, max=float(dp_vals[k]) * 2.0)
        dp_masked[{level_dim: k}] = xr.where(is_lowest_above, dp_adjusted, dp_masked[{level_dim: k}])

    return dp_masked


def compute_nwet_zwd(q: xr.DataArray, t: xr.DataArray, dp: xr.DataArray,
                     pressure_pa: np.ndarray, dim: str = "level") -> xr.DataArray:
    """
    Compute ZWD [mm] via direct per-level wet refractivity integration
    in pressure coordinates.

    ZWD = 10⁻⁶ × ∫ N_wet × (Rd × Tv) / (g × p) dp

    This avoids the Tm approximation by computing N_wet at each level.

    Args:
        q: Specific humidity [kg/kg].
        t: Temperature [K].
        dp: Pressure layer thicknesses [Pa].
        pressure_pa: Pressure levels [Pa].
        dim: Level dimension name.

    Returns:
        ZWD in [mm].
    """
    p_da = xr.DataArray(pressure_pa, dims=[dim],
                        coords={dim: pressure_pa / 100.0})

    # Vapour pressure [hPa] for refractivity constants
    e_pa = (q * p_da) / (0.622 + 0.378 * q)
    e_hpa = e_pa / 100.0

    # Wet refractivity [N-units] (Rueger 2002 constants)
    # Use full k2 (not k2') for direct N_wet integration
    N_wet = K2 * (e_hpa / t) + K3 * (e_hpa / t ** 2)

    # Virtual temperature [K]
    t_v = t * (1.0 + 0.608 * q)

    # Integrand: N_wet × (Rd × Tv) / (g × p) × dp
    # Units: [1] × [J/(kg·K)] × [K] / ([m/s²] × [Pa]) × [Pa]
    #       = [1] × [m] = [m]  (after 10⁻⁶ factor)
    integrand = N_wet * (R_D * t_v) / (G * p_da) * dp

    zwd_m = 1e-6 * integrand.sum(dim=dim)
    return zwd_m * 1000.0  # [mm]


# ============================================================
# INTEGRATION PIPELINE
# ============================================================
def compute_zwd_integrated(ds: xr.Dataset, every_n: int = 1,
                           label: str = "",
                           z_sfc: xr.DataArray = None) -> xr.DataArray:
    """
    Compute ZWD [mm] via vertical integration of q and T profiles.

    Args:
        ds: Dataset with specific_humidity, temperature, mean_sea_level_pressure.
        every_n: Subsample every N-th init_time.
        label: Label for logging.
        z_sfc: Surface geopotential [m²/s²]. If provided, MSLP is converted
               to actual surface pressure to properly mask underground levels.

    Returns:
        xr.DataArray of integrated ZWD in mm.
    """
    prefix = f"[{label}] " if label else ""

    if every_n > 1:
        n_total = ds.sizes["init_time"]
        idx = np.arange(0, n_total, every_n)
        ds = ds.isel(init_time=idx)
        print(f"  {prefix}Subsampled: {n_total} -> {len(idx)} init_times (every {every_n})")

    q  = ds["specific_humidity"]
    t  = ds["temperature"]
    mslp = ds["mean_sea_level_pressure"]

    levels = ds.level.values.astype(np.float64)
    pressure_pa = levels if levels.max() > 2000 else levels * 100.0

    mslp_sample = float(mslp.isel(init_time=0, lead_time=0).mean())
    if mslp_sample < 2000:
        print(f"  {prefix}Converting MSL from hPa to Pa (mean={mslp_sample:.0f})")
        mslp = mslp * 100.0

    # Estimate actual surface pressure from MSLP + surface geopotential
    if z_sfc is not None:
        t_sfc = ds["2m_temperature"] if "2m_temperature" in ds else None
        ps = estimate_surface_pressure(mslp, z_sfc, t_sfc=t_sfc)
        ps_mean = float(ps.isel(init_time=0, lead_time=0).mean())
        print(f"  {prefix}Using estimated surface pressure (mean={ps_mean:.0f} Pa)")
    else:
        ps = mslp
        print(f"  {prefix}WARNING: No surface geopotential — using MSLP as surface pressure.")
        print(f"  {prefix}         This overestimates ZWD over mountains!")

    print(f"  {prefix}Computing integration (direct N_wet)...", flush=True)
    dp  = compute_dp(pressure_pa, ps, level_dim="level")
    zwd = compute_nwet_zwd(q, t, dp, pressure_pa, dim="level")

    print(f"  {prefix}Loading into memory...", flush=True)
    zwd = zwd.compute()
    zwd.attrs = {"units": "mm", "long_name": f"ZWD integrated ({label})"}
    return zwd


# ============================================================
# STATISTICS
# ============================================================
def _compute_error_stats(source: xr.DataArray, target: xr.DataArray):
    """Compute bias, RMSE, STD, correlation between source and target."""
    diff = source - target
    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    std  = float(diff.std())
    corr = float(xr.corr(source, target))
    return {"bias": bias, "rmse": rmse, "std": std, "corr": corr}


def print_stats(sources: dict, zwd_target: xr.DataArray):
    """
    Print comparison statistics for all sources against target ZWD.

    Args:
        sources: dict of {label: xr.DataArray} for each ZWD source.
        zwd_target: target ground-truth ZWD.
    """
    print("\n" + "=" * 70)
    print("ZWD COMPARISON AGAINST TARGET")
    print("=" * 70)

    print(f"\n  Target ZWD: mean={float(zwd_target.mean()):.2f} mm, "
          f"std={float(zwd_target.std()):.2f} mm")

    for label, zwd in sources.items():
        zwd_aligned, tgt_aligned = xr.align(zwd, zwd_target, join="inner")
        stats = _compute_error_stats(zwd_aligned, tgt_aligned)
        print(f"\n  {label}:")
        print(f"    mean={float(zwd_aligned.mean()):.2f} mm, std={float(zwd_aligned.std()):.2f} mm")
        print(f"    Bias: {stats['bias']:+.2f} mm | RMSE: {stats['rmse']:.2f} mm | "
              f"STD: {stats['std']:.2f} mm | Corr: {stats['corr']:.4f}")

    # Per-timestep table
    spatial = ["latitude", "longitude"]
    ref_source_label = list(sources.keys())[0]
    ref_source = sources[ref_source_label]
    ref_aligned, tgt_aligned = xr.align(ref_source, zwd_target, join="inner")
    diff = ref_aligned - tgt_aligned

    n_init = diff.sizes.get("init_time", 1)
    n_lead = diff.sizes.get("lead_time", 1)
    n_total = n_init * n_lead

    if n_total > 1:
        # Build per-timestep stats for all sources
        source_ts = {}
        for label, zwd in sources.items():
            zwd_al, tgt_al = xr.align(zwd, zwd_target, join="inner")
            d = zwd_al - tgt_al
            source_ts[label] = {
                "bias": d.mean(dim=spatial),
                "rmse": np.sqrt((d ** 2).mean(dim=spatial)),
            }

        labels_short = [l[:20] for l in sources.keys()]
        header_parts = [f"{'init_time':<22s} {'lead_time':>9s}"]
        for ls in labels_short:
            header_parts.append(f"{'Bias(':>7s}{ls}{')':<1s}")
            header_parts.append(f"{'RMSE(':>7s}{ls}{')':<1s}")
        print(f"\n  {'  '.join(header_parts)}")
        print("  " + "-" * (30 + 20 * len(sources)))

        max_rows = 50
        show_all = n_total <= max_rows

        for ii in range(n_init):
            for li in range(n_lead):
                flat_idx = ii * n_lead + li
                if not show_all and max_rows // 2 <= flat_idx < n_total - max_rows // 2:
                    if flat_idx == max_rows // 2:
                        print(f"  ... ({n_total - max_rows} rows omitted) ...")
                    continue

                init_label = str(diff.init_time.values[ii])[:16]
                lead_label = str(diff.lead_time.values[li])
                parts = [f"  {init_label:<22s} {lead_label:>9s}"]
                for label in sources.keys():
                    b = float(source_ts[label]["bias"].isel(init_time=ii, lead_time=li))
                    r = float(source_ts[label]["rmse"].isel(init_time=ii, lead_time=li))
                    parts.append(f"  {b:>+8.2f}  {r:>8.2f}")
                print("".join(parts))

    print("=" * 70)


# ============================================================
# PLOTTING
# ============================================================
def _add_map_features(ax):
    """Add coastlines and borders to a cartopy GeoAxes."""
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4")



def _plot_map(data: xr.DataArray, fig, subplot_spec, cmap: str, title: str,
              vmin=None, vmax=None, symmetric=False):
    """
    Plot a 2D lat/lon field on a map with coastlines.

    Args:
        data: 2D DataArray with latitude/longitude coords.
        fig: matplotlib Figure.
        subplot_spec: subplot position (e.g., gs[0, 1]).
        cmap: colormap name.
        title: plot title.
        vmin, vmax: color limits.
        symmetric: if True, set vmin=-vmax centered on 0.
    """
    ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())

    if symmetric and vmax is not None:
        vmin = -vmax

    lats = data.latitude.values if "latitude" in data.coords else data.coords[list(data.dims)[0]].values
    lons = data.longitude.values if "longitude" in data.coords else data.coords[list(data.dims)[1]].values
    # Coarsen for bias-scale maps to avoid Moiré (only when grid is fine)
    plot_data = data
    if symmetric and len(lats) > 360:
        plot_data = data.coarsen(latitude=2, longitude=2, boundary="trim").mean()
        lats = plot_data.latitude.values
        lons = plot_data.longitude.values
    im = ax.pcolormesh(lons, lats, plot_data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
    _add_map_features(ax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    return ax


def _subsample_scatter(x, y, max_pts=200_000):
    """Flatten, filter NaN, subsample for scatter plots."""
    xf, yf = x.ravel(), y.ravel()
    valid = np.isfinite(xf) & np.isfinite(yf)
    xv, yv = xf[valid], yf[valid]
    if len(xv) > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xv), max_pts, replace=False)
        xv, yv = xv[idx], yv[idx]
    return xv, yv


def plot_per_source(label: str, zwd_source: xr.DataArray, zwd_target: xr.DataArray,
                    save_dir: str):
    """
    Per-source diagnostic plots against target: mean bias map, RMSE map,
    time series, scatter.  Maps use cartopy with coastlines.
    """
    from matplotlib.gridspec import GridSpec

    zwd_src, zwd_tgt = xr.align(zwd_source, zwd_target, join="inner")
    diff = zwd_src - zwd_tgt

    time_dims = [d for d in diff.dims if d in ("init_time", "lead_time")]
    spatial_dims = [d for d in diff.dims if d in ("latitude", "longitude")]

    mean_src_map  = zwd_src.mean(dim=time_dims)
    mean_tgt_map  = zwd_tgt.mean(dim=time_dims)
    mean_bias_map = diff.mean(dim=time_dims)
    rmse_map      = np.sqrt((diff ** 2).mean(dim=time_dims))

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle(f"{label} vs Target — Mean over all timesteps", fontsize=14)

    # Row 0: maps with coastlines
    _plot_map(mean_tgt_map, fig, gs[0, 0], cmap="Blues", title="Target ZWD [mm]")
    _plot_map(mean_src_map, fig, gs[0, 1], cmap="Blues", title=f"{label} [mm]")

    vmax_bias = max(abs(float(mean_bias_map.quantile(0.02))),
                    abs(float(mean_bias_map.quantile(0.98))), 0.1)
    _plot_map(mean_bias_map, fig, gs[0, 2], cmap="RdBu_r",
              title="Bias (source - target) [mm]", vmax=vmax_bias, symmetric=True)

    _plot_map(rmse_map, fig, gs[1, 0], cmap="Reds", title="RMSE [mm]")

    # Time series (regular axes)
    ax_ts = fig.add_subplot(gs[1, 1])
    bias_ts = diff.mean(dim=spatial_dims).values.ravel()
    step_idx = np.arange(len(bias_ts))
    ax_ts.bar(step_idx, bias_ts, color="steelblue", alpha=0.7)
    ax_ts.axhline(0, color="k", lw=0.5)
    ax_ts.set_xlabel("Timestep index")
    ax_ts.set_ylabel("Bias [mm]")
    ax_ts.set_title("Spatial-mean Bias per timestep")

    # Scatter (regular axes)
    ax_sc = fig.add_subplot(gs[1, 2])
    tgt_v, src_v = _subsample_scatter(zwd_tgt.values, zwd_src.values)
    ax_sc.scatter(tgt_v, src_v, s=0.5, alpha=0.1)
    lims = [min(tgt_v.min(), src_v.min()), max(tgt_v.max(), src_v.max())]
    ax_sc.plot(lims, lims, "r--", lw=1, label="1:1")
    ax_sc.set_xlabel("Target ZWD [mm]")
    ax_sc.set_ylabel(f"{label} [mm]")
    ax_sc.set_title("Scatter")
    ax_sc.legend()

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_")
    fname = os.path.join(save_dir, f"zwd_{safe_label}_vs_target.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_model_comparison(sources: dict, zwd_target: xr.DataArray, save_dir: str):
    """
    Combined plot comparing all sources against target.

    Layout:
      Row 0: bar chart | overlay scatter | RMSE difference map (with coastlines)
      Row 1: RMSE time series | bias time series
      Row 2: per-source mean bias maps (with coastlines)
    """
    from matplotlib.gridspec import GridSpec

    labels = list(sources.keys())
    n_sources = len(labels)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_sources]

    # Precompute aligned diffs
    aligned = {}
    for label, zwd in sources.items():
        src_al, tgt_al = xr.align(zwd, zwd_target, join="inner")
        aligned[label] = {"src": src_al, "tgt": tgt_al, "diff": src_al - tgt_al}

    time_dims = [d for d in zwd_target.dims if d in ("init_time", "lead_time")]
    spatial_dims = [d for d in zwd_target.dims if d in ("latitude", "longitude")]

    # Overall stats per source
    stats = {}
    for label in labels:
        d = aligned[label]["diff"]
        stats[label] = {
            "bias": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
        }

    n_cols = max(3, n_sources)
    fig = plt.figure(figsize=(6 * n_cols, 15))
    gs = GridSpec(3, n_cols, figure=fig)
    fig.suptitle("Model Comparison — All sources vs Target ZWD", fontsize=14)

    # --- Row 0, Col 0: Bar chart of overall Bias & RMSE ---
    ax_bar = fig.add_subplot(gs[0, 0])
    x = np.arange(n_sources)
    width = 0.35
    biases = [stats[l]["bias"] for l in labels]
    rmses  = [stats[l]["rmse"] for l in labels]
    ax_bar.bar(x - width/2, biases, width, color=colors, alpha=0.7, label="Bias")
    ax_bar.bar(x + width/2, rmses,  width, color=colors, alpha=0.4,
               edgecolor=colors, linewidth=1.5, label="RMSE")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_bar.axhline(0, color="k", lw=0.5)
    ax_bar.set_ylabel("[mm]")
    ax_bar.set_title("Overall Bias & RMSE")
    ax_bar.legend()

    # --- Row 0, Col 1: Overlay scatter ---
    ax_sc = fig.add_subplot(gs[0, 1])
    all_min, all_max = np.inf, -np.inf
    for i, label in enumerate(labels):
        tgt_v, src_v = _subsample_scatter(
            aligned[label]["tgt"].values, aligned[label]["src"].values,
            max_pts=100_000)
        ax_sc.scatter(tgt_v, src_v, s=0.3, alpha=0.1, color=colors[i], label=label)
        all_min = min(all_min, tgt_v.min(), src_v.min())
        all_max = max(all_max, tgt_v.max(), src_v.max())
    ax_sc.plot([all_min, all_max], [all_min, all_max], "r--", lw=1, label="1:1")
    ax_sc.set_xlabel("Target ZWD [mm]")
    ax_sc.set_ylabel("Source ZWD [mm]")
    ax_sc.set_title("Scatter (all sources)")
    ax_sc.legend(markerscale=5, fontsize=8)

    # --- Row 0, Col 2: RMSE difference map with coastlines ---
    if n_sources >= 2:
        rmse_0 = np.sqrt((aligned[labels[0]]["diff"] ** 2).mean(dim=time_dims))
        rmse_1 = np.sqrt((aligned[labels[1]]["diff"] ** 2).mean(dim=time_dims))
        rmse_diff = rmse_1 - rmse_0  # positive = first source is better
        vmax = max(abs(float(rmse_diff.quantile(0.02))),
                   abs(float(rmse_diff.quantile(0.98))), 0.1)
        _plot_map(rmse_diff, fig, gs[0, 2], cmap="RdBu_r",
                  title=f"RMSE({labels[1]}) - RMSE({labels[0]}) [mm]\n"
                        f"(blue = {labels[0]} better)",
                  vmax=vmax, symmetric=True)

    # --- Row 1, Col 0: RMSE time series overlaid ---
    ax_ts = fig.add_subplot(gs[1, 0])
    for i, label in enumerate(labels):
        d = aligned[label]["diff"]
        rmse_ts = np.sqrt((d ** 2).mean(dim=spatial_dims)).values.ravel()
        step_idx = np.arange(len(rmse_ts))
        ax_ts.plot(step_idx, rmse_ts, color=colors[i], lw=1.2, label=label, alpha=0.8)
    ax_ts.set_xlabel("Timestep index")
    ax_ts.set_ylabel("RMSE [mm]")
    ax_ts.set_title("Spatial-mean RMSE per timestep")
    ax_ts.legend(fontsize=9)

    # --- Row 1, Col 1: Bias time series overlaid ---
    ax_bias_ts = fig.add_subplot(gs[1, 1])
    for i, label in enumerate(labels):
        d = aligned[label]["diff"]
        bias_ts = d.mean(dim=spatial_dims).values.ravel()
        step_idx = np.arange(len(bias_ts))
        ax_bias_ts.plot(step_idx, bias_ts, color=colors[i], lw=1.2, label=label, alpha=0.8)
    ax_bias_ts.axhline(0, color="k", lw=0.5)
    ax_bias_ts.set_xlabel("Timestep index")
    ax_bias_ts.set_ylabel("Bias [mm]")
    ax_bias_ts.set_title("Spatial-mean Bias per timestep")
    ax_bias_ts.legend(fontsize=9)

    # --- Row 2: Per-source mean bias maps with coastlines ---
    for i, label in enumerate(labels):
        if i >= n_cols:
            break
        mean_bias = aligned[label]["diff"].mean(dim=time_dims)
        vmax = max(abs(float(mean_bias.quantile(0.02))),
                   abs(float(mean_bias.quantile(0.98))), 0.1)
        _plot_map(mean_bias, fig, gs[2, i], cmap="RdBu_r",
                  title=f"Mean Bias: {label} [mm]", vmax=vmax, symmetric=True)

    plt.tight_layout()
    fname = os.path.join(save_dir, "zwd_model_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare ZWD from Aurora models against target ground truth")
    parser.add_argument("--pred", required=True, help="Path to prediction zarr")
    parser.add_argument("--baseline", default=None,
                        help="Path to baseline zarr (no ZWD pred, integration only)")
    parser.add_argument("--target", required=True, help="Path to target zarr")
    parser.add_argument("--output", default=None, help="Save output dataset to zarr")
    parser.add_argument("--plot-dir", default="check_zwd_output", help="Directory to save plots")
    parser.add_argument("--every-n", type=int, default=1,
                        help="Use every N-th init_time step (default: all)")
    parser.add_argument("--static", default=None,
                        help="Path to static zarr/netcdf with geopotential_at_surface "
                             "(variable 'z' or 'geopotential_at_surface'). "
                             "Used to convert MSLP to actual surface pressure.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    # ---- Load surface geopotential (optional but recommended) ----
    z_sfc = None
    if args.static:
        print(f"\nLoading static data: {args.static}", flush=True)
        if args.static.endswith(".nc"):
            static_ds = xr.open_dataset(args.static)
        else:
            static_ds = xr.open_zarr(args.static)
        # Try common variable names
        for vname in ["geopotential_at_surface", "z", "orography"]:
            if vname in static_ds:
                z_sfc = static_ds[vname]
                # z might have a time dimension — take first timestep
                for dim in ["time", "init_time"]:
                    if dim in z_sfc.dims:
                        z_sfc = z_sfc.isel({dim: 0})
                z_sfc = z_sfc.compute()
                z_val = float(z_sfc.mean())
                print(f"  Loaded '{vname}': mean={z_val:.0f} m²/s² "
                      f"(≈ {z_val/G:.0f} m elevation)")
                break
        if z_sfc is None:
            print(f"  WARNING: No geopotential variable found in {args.static}. "
                  f"Available: {list(static_ds.data_vars)}")
    else:
        print("\nNote: No --static provided. MSLP will be used as surface pressure.")
        print("      For better accuracy over mountains, provide surface geopotential.")

    # ---- Load target and extract ground truth ZWD ----
    print(f"\n{'='*60}")
    print(f"Loading Target zarr: {args.target}", flush=True)
    target_ds = xr.open_zarr(args.target)
    print(f"  Variables : {list(target_ds.data_vars)}")
    print(f"  Dimensions: {dict(target_ds.dims)}")

    _, has_zwd_t = check_structure(target_ds, label="Target")
    if not has_zwd_t:
        print("ERROR: Target zarr must contain zenith_wet_delay. Exiting.")
        return

    zwd_target = target_ds["zenith_wet_delay"]
    if args.every_n > 1:
        n_total = target_ds.sizes["init_time"]
        idx = np.arange(0, n_total, args.every_n)
        zwd_target = zwd_target.isel(init_time=idx)
    zwd_target = zwd_target.compute()
    print(f"  Target ZWD: mean={float(zwd_target.mean()):.2f} mm, "
          f"range=[{float(zwd_target.min()):.1f}, {float(zwd_target.max()):.1f}] mm")

    # ---- Process prediction ----
    sources = {}

    print(f"\n{'='*60}")
    print(f"Loading Prediction zarr: {args.pred}", flush=True)
    pred_ds = xr.open_zarr(args.pred)
    print(f"  Variables : {list(pred_ds.data_vars)}")
    print(f"  Dimensions: {dict(pred_ds.dims)}")

    ok_p, has_zwd_p = check_structure(pred_ds, label="Prediction")
    if not ok_p:
        print("ERROR: Prediction zarr missing required atmos variables. Exiting.")
        return

    zwd_pred_int = compute_zwd_integrated(pred_ds, every_n=args.every_n,
                                          label="Finetuned integrated",
                                          z_sfc=z_sfc)
    sources["Finetuned integrated"] = zwd_pred_int

    if has_zwd_p:
        zwd_pred_direct = pred_ds["zenith_wet_delay"]
        if args.every_n > 1:
            idx = np.arange(0, pred_ds.sizes["init_time"], args.every_n)
            zwd_pred_direct = zwd_pred_direct.isel(init_time=idx)
        zwd_pred_direct = zwd_pred_direct.compute()
        sources["Finetuned predicted"] = zwd_pred_direct

    pred_ds.close()

    # ---- Process baseline (optional) ----
    if args.baseline:
        print(f"\n{'='*60}")
        print(f"Loading Baseline zarr: {args.baseline}", flush=True)
        base_ds = xr.open_zarr(args.baseline)
        print(f"  Variables : {list(base_ds.data_vars)}")
        print(f"  Dimensions: {dict(base_ds.dims)}")

        ok_b, _ = check_structure(base_ds, label="Baseline")
        if ok_b:
            zwd_base_int = compute_zwd_integrated(base_ds, every_n=args.every_n,
                                                  label="Baseline integrated",
                                                  z_sfc=z_sfc)
            sources["Baseline integrated"] = zwd_base_int
        else:
            print("WARNING: Baseline missing required atmos variables. Skipping.")
        base_ds.close()

    target_ds.close()

    # ---- Statistics ----
    print_stats(sources, zwd_target)

    # ---- Plots ----
    if not args.no_plot:
        # Per-source plots
        for label, zwd in sources.items():
            plot_per_source(label, zwd, zwd_target, save_dir=args.plot_dir)

        # Combined comparison plot
        plot_model_comparison(sources, zwd_target, save_dir=args.plot_dir)

    # ---- Save output ----
    if args.output:
        ds_out = xr.Dataset({"zwd_target_mm": zwd_target})
        for label, zwd in sources.items():
            safe = label.lower().replace(" ", "_")
            ds_out[f"zwd_{safe}_mm"] = zwd
        print(f"\nSaving to {args.output}...", flush=True)
        ds_out.to_zarr(args.output, mode="w")
        print("Done.")


if __name__ == "__main__":
    main()
