"""Connected-component gap analysis for LiDAR change-detection point clouds.

This module takes C2C (cloud-to-cloud) distance arrays already computed by
the main pipeline and:
  1. Verifies spatial overlap between epoch clouds and clips if needed
  2. Extracts connected components from thresholded damage / recovery masks
  3. Computes per-component statistics (area, centroid, C2C moments)
  4. Fits power-law gap-size distributions (MLE + log-log OLS)
  5. Produces diagnostic figures and epoch-comparison summaries

All heavy geometry lives in cloudComPy; statistics use only numpy / scipy.

References
----------
- Clauset, Shalizi & Newman (2009) — MLE power-law fitting
- Leitold et al. — 2 m C2C damage threshold
- Gorgens et al. — 10 m^2 minimum detectable gap area
"""

from __future__ import annotations

import logging
from math import log2, sqrt
from pathlib import Path
from typing import Any

import cloudComPy as cc
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / VNC sessions
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "damage_threshold": 2.0,       # metres (Leitold et al.)
    "recovery_threshold": 2.0,     # metres
    "min_component_pts": 100,
    "min_gap_area_m2": 10.0,       # Gorgens et al.
    "overlap_min_fraction": 0.80,
    "overlap_buffer_m": 50.0,
    "max_components": 100_000,
}


def _cfg(config: dict | None, key: str) -> Any:
    """Return *config[key]* with a fallback to DEFAULT_CONFIG."""
    if config and key in config:
        return config[key]
    return DEFAULT_CONFIG[key]


# ===================================================================
# Task 1 — Spatial Overlap Verification & Clipping
# ===================================================================

def _bb_xy(cloud) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) from a cloud's own bounding box."""
    bb = cloud.getOwnBB()
    lo = bb.minCorner()
    hi = bb.maxCorner()
    return (lo[0], lo[1], hi[0], hi[1])


def _intersect_xy(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Compute the XY intersection of axis-aligned bounding boxes.

    Returns None when the intersection is empty.
    """
    xmin = max(b[0] for b in boxes)
    ymin = max(b[1] for b in boxes)
    xmax = min(b[2] for b in boxes)
    ymax = min(b[3] for b in boxes)
    if xmin >= xmax or ymin >= ymax:
        return None
    return (xmin, ymin, xmax, ymax)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def verify_and_clip_overlap(
    clouds_dict: dict[str, Any],
    label: str,
    buffer_m: float = 50.0,
    config: dict | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check XY overlap among epoch clouds and clip if partial.

    Parameters
    ----------
    clouds_dict : dict
        Mapping ``{"2017": cloud, "2018": cloud, "2020": cloud}``.
    label : str
        Grid / tile identifier for log messages.
    buffer_m : float
        Metres to expand the intersection box before clipping.
    config : dict, optional
        Pipeline configuration (uses *overlap_min_fraction*).

    Returns
    -------
    clipped_clouds : dict
        Same keys as *clouds_dict*, values may be clipped copies.
    overlap_report : dict
        Diagnostic information about the overlap test.

    Raises
    ------
    ValueError
        If the XY intersection is empty (clouds do not overlap at all).
    """
    min_fraction = _cfg(config, "overlap_min_fraction")

    # Gather per-cloud XY boxes and areas
    names = sorted(clouds_dict.keys())
    boxes: dict[str, tuple[float, float, float, float]] = {}
    areas: dict[str, float] = {}
    for name in names:
        b = _bb_xy(clouds_dict[name])
        boxes[name] = b
        areas[name] = _box_area(b)

    intersection = _intersect_xy(list(boxes.values()))

    if intersection is None:
        msg = (
            f"[OVERLAP] {label}: clouds have NO XY overlap — "
            "cannot run gap analysis on this grid."
        )
        print(msg)
        raise ValueError(msg)

    inter_area = _box_area(intersection)
    smallest_area = min(areas.values())
    overlap_frac = inter_area / smallest_area if smallest_area > 0 else 0.0

    report: dict[str, Any] = {
        "intersection_area_m2": inter_area,
        "cloud_areas": dict(areas),
        "overlap_fraction": overlap_frac,
        "was_clipped": False,
        "buffer_used": 0.0,
    }

    if overlap_frac >= min_fraction:
        print(
            f"[OVERLAP] {label}: overlap {overlap_frac:.1%} of smallest cloud "
            f"({inter_area:.0f} m^2) — no clipping needed."
        )
        return dict(clouds_dict), report

    # --- Partial overlap → clip to intersection + buffer ---
    print(
        f"[OVERLAP] {label}: overlap only {overlap_frac:.1%} — "
        f"clipping all clouds to intersection + {buffer_m} m buffer."
    )

    # Retrieve global Z range across all clouds so we don't accidentally
    # remove points in Z.
    z_min = min(clouds_dict[n].getOwnBB().minCorner()[2] for n in names)
    z_max = max(clouds_dict[n].getOwnBB().maxCorner()[2] for n in names)

    clip_box = cc.ccBBox(
        (intersection[0] - buffer_m, intersection[1] - buffer_m, z_min - 1.0),
        (intersection[2] + buffer_m, intersection[3] + buffer_m, z_max + 1.0),
        True,
    )

    clipped: dict[str, Any] = {}
    for name in names:
        cropped = cc.cropPointCloud(clouds_dict[name], clip_box, inside=True)
        if cropped is None or cropped.size() == 0:
            print(
                f"[OVERLAP] {label}: WARNING — cloud {name} is empty after "
                "clipping; falling back to unclipped cloud."
            )
            clipped[name] = clouds_dict[name]
        else:
            clipped[name] = cropped

    report["was_clipped"] = True
    report["buffer_used"] = buffer_m
    return clipped, report


# ===================================================================
# Task 2 — Connected Components Extraction
# ===================================================================

def _dynamic_octree_level(cloud) -> int:
    """Choose an octree level so that the cell size is roughly 2 m.

    The cell side length at level *L* of an octree spanning diagonal *D*
    is approximately ``D / 2^L``.  We want cell ≈ 2 m ⇒ L ≈ log2(D/2).
    """
    bb = cloud.getOwnBB()
    lo, hi = bb.minCorner(), bb.maxCorner()
    diag = sqrt(
        (hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2 + (hi[2] - lo[2]) ** 2
    )
    if diag <= 0:
        return 8  # safe fallback
    level = int(log2(max(diag / 2.0, 1.0)))
    return max(6, min(12, level))


def extract_damage_components(
    source_cloud,
    c2c_array: np.ndarray,
    sf_name: str,
    grid_id: str,
    epoch_label: str,
    config: dict | None = None,
) -> tuple[list, list, int]:
    """Threshold a C2C array and extract connected components.

    Parameters
    ----------
    source_cloud : ccPointCloud
        The reference cloud whose geometry is used.
    c2c_array : np.ndarray
        Float64 C2C distance per point (same ordering as *source_cloud*).
    sf_name : str
        Name that was (or will be) used for the scalar field.
    grid_id : str
        Grid / tile identifier for log messages.
    epoch_label : str
        ``"damage_1718"`` or ``"recovery_1820"``.
    config : dict, optional
        Must contain *damage_threshold*, *min_component_pts*.

    Returns
    -------
    components : list[ccPointCloud]
        Connected-component clouds (caller must delete them).
    residuals : list[ccPointCloud]
        Residual (too-small) component clouds (caller must delete them).
    octree_level : int
        The octree level that was used.
    """
    threshold = _cfg(config, "damage_threshold")
    min_pts = _cfg(config, "min_component_pts")
    max_comps = _cfg(config, "max_components")

    mask = c2c_array > threshold
    indices = np.where(mask)[0].astype(np.int32)

    if indices.size == 0:
        print(
            f"[COMP] {grid_id}/{epoch_label}: no points exceed "
            f"{threshold} m threshold — skipping component extraction."
        )
        return [], [], 0

    print(
        f"[COMP] {grid_id}/{epoch_label}: {indices.size} / {c2c_array.size} "
        f"points ({100.0 * indices.size / c2c_array.size:.1f}%) above "
        f"{threshold} m threshold."
    )

    # --- Create partial clone of thresholded points ---
    ref = cc.ReferenceCloud(source_cloud)
    for idx in indices:
        ref.addPointIndex(int(idx))
    damaged_cloud, res_code = source_cloud.partialClone(ref)
    if res_code != 0 or damaged_cloud is None:
        print(
            f"[COMP] {grid_id}/{epoch_label}: WARNING — partialClone failed "
            f"(code={res_code}). Returning empty components."
        )
        return [], [], 0

    # Attach C2C values as a scalar field on the cloned cloud so that
    # downstream stats can read them directly from each component.
    sf_idx = damaged_cloud.addScalarField(sf_name)
    if sf_idx >= 0:
        sf = damaged_cloud.getScalarField(sf_idx)
        masked_vals = c2c_array[mask]
        np_sf = sf.toNpArrayCopy()
        # The SF array is the same length as the cloned cloud.
        np.copyto(np_sf, masked_vals[:np_sf.size])
        # Write back: iterate to set each value (toNpArrayCopy is a copy).
        for i in range(int(damaged_cloud.size())):
            sf.setValue(i, float(masked_vals[i]))
        damaged_cloud.setCurrentDisplayedScalarField(sf_idx)

    # --- Choose octree level ---
    octree_level = _dynamic_octree_level(damaged_cloud)
    print(
        f"[COMP] {grid_id}/{epoch_label}: using octree level {octree_level} "
        f"for component extraction."
    )

    # --- Extract connected components ---
    n_processed, components, residuals = cc.ExtractConnectedComponents(
        clouds=[damaged_cloud],
        octreeLevel=octree_level,
        minComponentSize=min_pts,
        maxNumberComponents=max_comps,
        randomColors=False,
    )

    print(
        f"[COMP] {grid_id}/{epoch_label}: extracted {len(components)} "
        f"components, {len(residuals)} residual clouds "
        f"(octreeLevel={octree_level})."
    )

    # The intermediate damaged_cloud is no longer needed — the components
    # hold their own geometry.  Delete to free memory.
    cc.deleteEntity(damaged_cloud)

    return components, residuals, octree_level


# ===================================================================
# Task 3 — Per-Component Statistics
# ===================================================================

def _sf_array_from_cloud(cloud) -> np.ndarray | None:
    """Try to extract the first scalar field as a numpy array."""
    sf_dict = cloud.getScalarFieldDic()
    if not sf_dict:
        return None
    first_name = next(iter(sf_dict))
    sf = cloud.getScalarField(first_name)
    if sf is None:
        return None
    return sf.toNpArrayCopy()


def compute_component_stats(
    components: list,
    residuals: list,
    grid_id: str,
    epoch_label: str,
    point_density_per_m2: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute descriptive statistics for each connected component.

    Parameters
    ----------
    components : list[ccPointCloud]
        Component clouds from ``ExtractConnectedComponents``.
    residuals : list[ccPointCloud]
        Residual clouds (too-small components).
    grid_id, epoch_label : str
        Identifiers propagated into every record.
    point_density_per_m2 : float, optional
        If provided, an area estimate is computed as ``n / density``.

    Returns
    -------
    stats_list : list[dict]
        One dict per component.
    residual_summary : dict
        Aggregate information about residual clouds.
    """
    stats_list: list[dict[str, Any]] = []

    for comp_idx, comp in enumerate(components):
        n_pts = int(comp.size())
        bb = comp.getOwnBB()
        lo, hi = bb.minCorner(), bb.maxCorner()
        width = hi[0] - lo[0]
        height = hi[1] - lo[1]
        bbox_area = width * height
        centroid_x = 0.5 * (lo[0] + hi[0])
        centroid_y = 0.5 * (lo[1] + hi[1])

        # C2C scalar-field moments
        sf_arr = _sf_array_from_cloud(comp)
        if sf_arr is not None and sf_arr.size > 0:
            mean_c2c = float(np.mean(sf_arr))
            median_c2c = float(np.median(sf_arr))
            std_c2c = float(np.std(sf_arr))
            max_c2c = float(np.max(sf_arr))
        else:
            mean_c2c = median_c2c = std_c2c = max_c2c = float("nan")

        record: dict[str, Any] = {
            "grid_id": grid_id,
            "epoch_label": epoch_label,
            "component_id": comp_idx,
            "n_points": n_pts,
            "bbox_area_m2": bbox_area,
            "bbox_area_sqrt_m": sqrt(max(bbox_area, 0.0)),
            "mean_c2c_m": mean_c2c,
            "median_c2c_m": median_c2c,
            "std_c2c_m": std_c2c,
            "max_c2c_m": max_c2c,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
        }

        if point_density_per_m2 is not None and point_density_per_m2 > 0:
            record["estimated_area_m2"] = n_pts / point_density_per_m2

        stats_list.append(record)

    # --- Residual summary ---
    total_residual_pts = sum(int(r.size()) for r in residuals)
    residual_summary: dict[str, Any] = {
        "grid_id": grid_id,
        "epoch_label": epoch_label,
        "n_residual_clouds": len(residuals),
        "total_residual_points": total_residual_pts,
    }

    print(
        f"[COMP] {grid_id}/{epoch_label}: computed stats for "
        f"{len(stats_list)} components, {len(residuals)} residual clouds "
        f"({total_residual_pts} residual points)."
    )

    return stats_list, residual_summary


# ===================================================================
# Task 4 — Power Law Analysis
# ===================================================================

def fit_power_law(
    areas: np.ndarray,
    label: str = "",
    min_area_m2: float = 10.0,
) -> dict[str, Any]:
    """Fit a power-law distribution to gap areas.

    Two estimators are computed:

    1. **MLE** (Clauset et al. 2009):
       ``alpha = 1 + n / sum(ln(A_i / A_min))``
    2. **Log-log OLS**: logarithmic binning, linear regression on
       ``log(count)`` vs ``log(bin_centre)``.

    Goodness of fit is assessed via the KS statistic, log-likelihood,
    and a likelihood-ratio test against a log-normal alternative.

    Parameters
    ----------
    areas : array-like
        Gap areas in m^2 (unfiltered).
    label : str
        Label for log messages.
    min_area_m2 : float
        Minimum area for the fit (x_min in the power-law literature).

    Returns
    -------
    result : dict
        Keys: ``alpha_mle``, ``alpha_ols``, ``xmin``, ``n_fitted``,
        ``ks_stat``, ``ks_pvalue``, ``r2_loglog``, ``ll_powerlaw``,
        ``ll_lognormal``, ``preferred_distribution``, ``areas_fitted``.
    """
    areas = np.asarray(areas, dtype=np.float64)
    fitted = areas[areas >= min_area_m2]

    result: dict[str, Any] = {
        "alpha_mle": float("nan"),
        "alpha_ols": float("nan"),
        "xmin": min_area_m2,
        "n_fitted": 0,
        "ks_stat": float("nan"),
        "ks_pvalue": float("nan"),
        "r2_loglog": float("nan"),
        "ll_powerlaw": float("nan"),
        "ll_lognormal": float("nan"),
        "preferred_distribution": "undetermined",
        "areas_fitted": np.array([], dtype=np.float64),
    }

    if fitted.size < 5:
        print(
            f"[POWERLAW] {label}: only {fitted.size} areas >= {min_area_m2} m^2"
            " — too few for a reliable fit."
        )
        return result

    n = fitted.size
    xmin = min_area_m2
    result["n_fitted"] = n
    result["areas_fitted"] = fitted

    # ---- MLE ----
    log_ratios = np.log(fitted / xmin)
    sum_log = float(np.sum(log_ratios))
    if sum_log > 0:
        alpha_mle = 1.0 + n / sum_log
    else:
        alpha_mle = float("nan")
    result["alpha_mle"] = alpha_mle

    # ---- Log-log OLS ----
    n_bins = max(10, int(sqrt(n)))
    log_edges = np.linspace(np.log10(xmin), np.log10(fitted.max()), n_bins + 1)
    counts, _ = np.histogram(np.log10(fitted), bins=log_edges)
    bin_centres = 0.5 * (log_edges[:-1] + log_edges[1:])
    positive = counts > 0
    if positive.sum() >= 2:
        x_ols = bin_centres[positive]
        y_ols = np.log10(counts[positive].astype(np.float64))
        slope, intercept, r_value, _, _ = sp_stats.linregress(x_ols, y_ols)
        result["alpha_ols"] = -slope  # P(A) ~ A^{-alpha} → slope is negative
        result["r2_loglog"] = r_value ** 2
    else:
        result["alpha_ols"] = float("nan")
        result["r2_loglog"] = float("nan")

    # ---- KS statistic ----
    if np.isfinite(alpha_mle) and alpha_mle > 1.0:
        # Theoretical CDF: F(x) = 1 - (x / xmin)^{-(alpha-1)}
        sorted_a = np.sort(fitted)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = 1.0 - (sorted_a / xmin) ** (-(alpha_mle - 1.0))
        ks_stat = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))
        result["ks_stat"] = ks_stat
        # Approximate p-value via scipy's KS one-sample test against the
        # fitted Pareto.  scipy.stats.pareto uses shape a where PDF
        # ∝ x^{-(a+1)} with loc/scale, so a = alpha - 1, scale = xmin.
        ks_result = sp_stats.kstest(
            fitted,
            "pareto",
            args=(alpha_mle - 1.0,),
            N=n,
            alternative="two-sided",
        )
        result["ks_stat"] = float(ks_result.statistic)
        result["ks_pvalue"] = float(ks_result.pvalue)

    # ---- Log-likelihoods ----
    # Power-law: pdf = (alpha-1)/xmin * (x/xmin)^{-alpha}
    if np.isfinite(alpha_mle) and alpha_mle > 1.0:
        ll_pl = float(
            n * np.log(alpha_mle - 1.0)
            - n * np.log(xmin)
            - alpha_mle * np.sum(np.log(fitted / xmin))
        )
        result["ll_powerlaw"] = ll_pl
    else:
        ll_pl = -np.inf

    # Log-normal MLE for comparison
    log_fitted = np.log(fitted)
    mu_ln = float(np.mean(log_fitted))
    sigma_ln = float(np.std(log_fitted, ddof=1)) if n > 1 else 1.0
    if sigma_ln > 0:
        ll_ln = float(np.sum(sp_stats.lognorm.logpdf(
            fitted, s=sigma_ln, scale=np.exp(mu_ln)
        )))
    else:
        ll_ln = -np.inf
    result["ll_lognormal"] = ll_ln

    # Vuong-style likelihood-ratio test
    if np.isfinite(ll_pl) and np.isfinite(ll_ln):
        if ll_pl > ll_ln:
            result["preferred_distribution"] = "power_law"
        elif ll_ln > ll_pl:
            result["preferred_distribution"] = "lognormal"
        else:
            result["preferred_distribution"] = "indeterminate"
    elif np.isfinite(ll_pl):
        result["preferred_distribution"] = "power_law"
    elif np.isfinite(ll_ln):
        result["preferred_distribution"] = "lognormal"

    tag = f" ({label})" if label else ""
    print(
        f"[POWERLAW]{tag}: alpha_MLE={alpha_mle:.3f}, "
        f"alpha_OLS={result['alpha_ols']:.3f}, n={n}, "
        f"KS={result['ks_stat']:.4f}, preferred={result['preferred_distribution']}"
    )

    return result


def compare_power_laws(
    fit_a: dict[str, Any],
    fit_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Compare two power-law fits across epochs.

    Parameters
    ----------
    fit_a, fit_b : dict
        Outputs of ``fit_power_law``.
    label_a, label_b : str
        Human-readable epoch labels (e.g. ``"damage_1718"``).

    Returns
    -------
    comparison : dict
        Includes ``delta_alpha``, ``interpretation``, per-epoch summaries.
    """
    alpha_a = fit_a.get("alpha_mle", float("nan"))
    alpha_b = fit_b.get("alpha_mle", float("nan"))
    delta = alpha_a - alpha_b  # damage − recovery

    if np.isfinite(delta):
        if delta < 0:
            interp = (
                f"{label_a} has a flatter tail (alpha={alpha_a:.2f}) than "
                f"{label_b} (alpha={alpha_b:.2f}): the first epoch produced "
                "relatively more large gaps."
            )
        elif delta > 0:
            interp = (
                f"{label_b} has a flatter tail (alpha={alpha_b:.2f}) than "
                f"{label_a} (alpha={alpha_a:.2f}): the second epoch produced "
                "relatively more large gaps."
            )
        else:
            interp = "Both epochs have identical power-law exponents."
    else:
        interp = "Cannot compare — at least one fit is invalid."

    comparison = {
        "label_a": label_a,
        "label_b": label_b,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "delta_alpha": delta,
        "interpretation": interp,
    }

    print(
        f"[POWERLAW] comparison {label_a} vs {label_b}: "
        f"delta_alpha={delta:.3f}"
    )
    return comparison


# ===================================================================
# Task 5 — Gap Size Distribution Figure
# ===================================================================

def gap_size_distribution_analysis(
    component_stats: list[dict[str, Any]],
    grid_id: str,
    epoch_label: str,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Fit power law and produce a 4-panel diagnostic figure.

    Panels:
      (a) Log-log histogram with power-law overlay
      (b) Empirical vs theoretical CDF
      (c) Gap area vs mean C2C scatter
      (d) Zipf (rank-size) plot

    Parameters
    ----------
    component_stats : list[dict]
        Output of ``compute_component_stats``.
    grid_id, epoch_label : str
        Identifiers used in filenames and titles.
    out_dir : str or Path
        Directory for the saved figure.

    Returns
    -------
    fit : dict
        Output of ``fit_power_law``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    areas = np.array(
        [s["bbox_area_m2"] for s in component_stats], dtype=np.float64
    )
    mean_c2c = np.array(
        [s["mean_c2c_m"] for s in component_stats], dtype=np.float64
    )

    fit = fit_power_law(areas, label=f"{grid_id}/{epoch_label}")

    if areas.size == 0:
        print(
            f"[DIST] {grid_id}/{epoch_label}: no components — skipping figure."
        )
        return fit

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Gap Size Distribution — {grid_id} / {epoch_label}", fontsize=13
    )

    # ---- (a) Log-log histogram with power-law line ----
    ax = axes[0, 0]
    pos_areas = areas[areas > 0]
    if pos_areas.size > 0:
        log_bins = np.logspace(
            np.log10(pos_areas.min()),
            np.log10(pos_areas.max()),
            max(10, int(sqrt(pos_areas.size))),
        )
        ax.hist(pos_areas, bins=log_bins, edgecolor="k", alpha=0.7, label="data")
        alpha = fit["alpha_mle"]
        xmin = fit["xmin"]
        if np.isfinite(alpha) and alpha > 1.0:
            x_line = np.logspace(np.log10(xmin), np.log10(pos_areas.max()), 200)
            # Normalised PDF: p(x) = (alpha-1)/xmin * (x/xmin)^{-alpha}
            # Scale to match histogram counts
            bin_widths = np.diff(log_bins)
            total_in_range = np.sum(pos_areas >= xmin)
            pdf_line = (
                (alpha - 1.0) / xmin * (x_line / xmin) ** (-alpha)
            )
            # Scale: integral of pdf over each bin ≈ pdf * bin_width * N
            mean_bw = float(np.mean(bin_widths))
            scale_factor = total_in_range * mean_bw
            ax.plot(
                x_line, pdf_line * scale_factor,
                "r-", lw=2,
                label=f"PL fit (alpha={alpha:.2f})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap area (m$^2$)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Log-log histogram")
    ax.legend(fontsize=8)
    _annotate_fit(ax, fit)

    # ---- (b) Empirical vs theoretical CDF ----
    ax = axes[0, 1]
    sorted_a = np.sort(areas)
    ecdf = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    ax.step(sorted_a, ecdf, where="post", label="Empirical CDF", lw=1.5)
    alpha = fit["alpha_mle"]
    xmin = fit["xmin"]
    if np.isfinite(alpha) and alpha > 1.0 and xmin > 0:
        x_th = np.sort(sorted_a[sorted_a >= xmin])
        if x_th.size > 0:
            tcdf = 1.0 - (x_th / xmin) ** (-(alpha - 1.0))
            # Rescale theoretical CDF to start at the fraction below xmin
            frac_below = np.searchsorted(sorted_a, xmin) / len(sorted_a)
            tcdf_scaled = frac_below + (1.0 - frac_below) * tcdf
            ax.plot(x_th, tcdf_scaled, "r--", lw=1.5, label="Power-law CDF")
    ax.set_xscale("log")
    ax.set_xlabel("Gap area (m$^2$)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) CDF comparison")
    ax.legend(fontsize=8)

    # ---- (c) Area vs mean C2C ----
    ax = axes[1, 0]
    valid = np.isfinite(mean_c2c) & (areas > 0)
    if valid.any():
        ax.scatter(areas[valid], mean_c2c[valid], s=8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Gap area (m$^2$)")
    ax.set_ylabel("Mean C2C distance (m)")
    ax.set_title("(c) Gap area vs C2C displacement")

    # ---- (d) Rank-size (Zipf) plot ----
    ax = axes[1, 1]
    ranks = np.arange(1, len(sorted_a) + 1)
    ax.scatter(ranks, sorted_a[::-1], s=8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Gap area (m$^2$)")
    ax.set_title("(d) Rank-size (Zipf) plot")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = out_dir / f"{grid_id}_{epoch_label}_gap_distribution.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[DIST] {grid_id}/{epoch_label}: saved figure → {fname}")

    return fit


def _annotate_fit(ax, fit: dict[str, Any]) -> None:
    """Add a small text box with fit parameters to an axes."""
    alpha = fit.get("alpha_mle", float("nan"))
    r2 = fit.get("r2_loglog", float("nan"))
    n = fit.get("n_fitted", 0)
    xmin = fit.get("xmin", float("nan"))
    text = (
        f"$\\alpha_{{MLE}}$ = {alpha:.2f}\n"
        f"$R^2_{{OLS}}$ = {r2:.3f}\n"
        f"n = {n}\n"
        f"$x_{{min}}$ = {xmin:.0f} m$^2$"
    )
    ax.text(
        0.97, 0.97, text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )


# ===================================================================
# Task 6 — Epoch Comparison
# ===================================================================

def compare_epochs(
    stats_1718: list[dict[str, Any]],
    stats_1820: list[dict[str, Any]],
    grid_id: str,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Compare gap-size distributions between the two analysis epochs.

    Parameters
    ----------
    stats_1718, stats_1820 : list[dict]
        Component statistics from ``compute_component_stats``.
    grid_id : str
        Grid identifier.
    out_dir : str or Path
        Directory for saved figures.

    Returns
    -------
    comparison : dict
        Full comparison dictionary suitable for CSV export.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    areas_17 = np.array(
        [s["bbox_area_m2"] for s in stats_1718], dtype=np.float64
    )
    areas_18 = np.array(
        [s["bbox_area_m2"] for s in stats_1820], dtype=np.float64
    )

    fit_17 = fit_power_law(areas_17, label=f"{grid_id}/damage_1718")
    fit_18 = fit_power_law(areas_18, label=f"{grid_id}/recovery_1820")
    pl_cmp = compare_power_laws(fit_17, fit_18, "damage_1718", "recovery_1820")

    # --- Distribution tests ---
    ks_result: dict[str, Any] = {"statistic": float("nan"), "pvalue": float("nan")}
    mw_result: dict[str, Any] = {"statistic": float("nan"), "pvalue": float("nan")}
    if areas_17.size > 0 and areas_18.size > 0:
        ks = sp_stats.ks_2samp(areas_17, areas_18)
        ks_result = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
        mw = sp_stats.mannwhitneyu(areas_17, areas_18, alternative="two-sided")
        mw_result = {"statistic": float(mw.statistic), "pvalue": float(mw.pvalue)}

    # --- Summary metrics ---
    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size > 0 else 0.0

    def _safe_sum(arr: np.ndarray) -> float:
        return float(np.sum(arr)) if arr.size > 0 else 0.0

    # Size classes: <100, 100-500, >500 m^2
    def _size_class_counts(arr: np.ndarray) -> dict[str, int]:
        return {
            "lt_100": int(np.sum(arr < 100)),
            "100_to_500": int(np.sum((arr >= 100) & (arr < 500))),
            "gt_500": int(np.sum(arr >= 500)),
        }

    comparison: dict[str, Any] = {
        "grid_id": grid_id,
        "n_gaps_damage": int(areas_17.size),
        "n_gaps_recovery": int(areas_18.size),
        "gap_count_change": int(areas_18.size) - int(areas_17.size),
        "mean_gap_area_damage_m2": _safe_mean(areas_17),
        "mean_gap_area_recovery_m2": _safe_mean(areas_18),
        "mean_gap_size_change_m2": _safe_mean(areas_18) - _safe_mean(areas_17),
        "total_damaged_area_damage_m2": _safe_sum(areas_17),
        "total_damaged_area_recovery_m2": _safe_sum(areas_18),
        "total_area_change_m2": _safe_sum(areas_18) - _safe_sum(areas_17),
        "size_classes_damage": _size_class_counts(areas_17),
        "size_classes_recovery": _size_class_counts(areas_18),
        "ks_2samp": ks_result,
        "mann_whitney_u": mw_result,
        "power_law_damage": {
            k: v for k, v in fit_17.items() if k != "areas_fitted"
        },
        "power_law_recovery": {
            k: v for k, v in fit_18.items() if k != "areas_fitted"
        },
        "power_law_comparison": pl_cmp,
    }

    # --- Figure ---
    _plot_epoch_comparison(
        areas_17, areas_18, fit_17, fit_18, grid_id, out_dir,
    )

    return comparison


def _plot_epoch_comparison(
    areas_a: np.ndarray,
    areas_b: np.ndarray,
    fit_a: dict,
    fit_b: dict,
    grid_id: str,
    out_dir: Path,
) -> None:
    """Produce a 3-panel epoch comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Epoch Comparison — {grid_id}", fontsize=13)

    # ---- (a) Overlaid log-log histograms ----
    ax = axes[0]
    all_areas = np.concatenate([areas_a[areas_a > 0], areas_b[areas_b > 0]])
    if all_areas.size > 0:
        log_bins = np.logspace(
            np.log10(max(all_areas.min(), 0.1)),
            np.log10(all_areas.max()),
            30,
        )
        if areas_a[areas_a > 0].size > 0:
            ax.hist(
                areas_a[areas_a > 0], bins=log_bins, alpha=0.5,
                edgecolor="k", label="Damage 17-18",
            )
        if areas_b[areas_b > 0].size > 0:
            ax.hist(
                areas_b[areas_b > 0], bins=log_bins, alpha=0.5,
                edgecolor="k", label="Recovery 18-20",
            )
        # Power-law lines
        for fit, color, lbl in [
            (fit_a, "red", "PL damage"),
            (fit_b, "blue", "PL recovery"),
        ]:
            alpha = fit["alpha_mle"]
            xmin = fit["xmin"]
            if np.isfinite(alpha) and alpha > 1.0 and all_areas.max() > xmin:
                x_line = np.logspace(
                    np.log10(xmin), np.log10(all_areas.max()), 200
                )
                pdf_line = (
                    (alpha - 1.0) / xmin * (x_line / xmin) ** (-alpha)
                )
                # Rough scaling
                n_fit = fit["n_fitted"]
                mean_bw = float(np.mean(np.diff(log_bins)))
                ax.plot(
                    x_line,
                    pdf_line * n_fit * mean_bw,
                    color=color, lw=2,
                    label=f"{lbl} (a={alpha:.2f})",
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap area (m$^2$)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Overlaid histograms")
    ax.legend(fontsize=7)

    # ---- (b) Side-by-side CDFs ----
    ax = axes[1]
    for arr, lbl, color in [
        (areas_a, "Damage 17-18", "red"),
        (areas_b, "Recovery 18-20", "blue"),
    ]:
        if arr.size > 0:
            s = np.sort(arr)
            ecdf = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, ecdf, where="post", label=lbl, color=color, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Gap area (m$^2$)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) Empirical CDFs")
    ax.legend(fontsize=8)

    # ---- (c) Bar chart by size class ----
    ax = axes[2]
    classes = ["< 100", "100–500", "> 500"]
    counts_a = [
        int(np.sum(areas_a < 100)),
        int(np.sum((areas_a >= 100) & (areas_a < 500))),
        int(np.sum(areas_a >= 500)),
    ]
    counts_b = [
        int(np.sum(areas_b < 100)),
        int(np.sum((areas_b >= 100) & (areas_b < 500))),
        int(np.sum(areas_b >= 500)),
    ]
    x_pos = np.arange(len(classes))
    width = 0.35
    ax.bar(x_pos - width / 2, counts_a, width, label="Damage 17-18", alpha=0.8)
    ax.bar(x_pos + width / 2, counts_b, width, label="Recovery 18-20", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Gap area class (m$^2$)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Gap count by size class")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = out_dir / f"{grid_id}_epoch_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[DIST] {grid_id}: saved epoch comparison → {fname}")


# ===================================================================
# Task 7 — Main Entry Point
# ===================================================================

def _estimate_point_density(cloud) -> float:
    """Estimate planar point density (pts / m^2) from a cloud's XY bbox."""
    bb = cloud.getOwnBB()
    lo, hi = bb.minCorner(), bb.maxCorner()
    width = hi[0] - lo[0]
    height = hi[1] - lo[1]
    xy_area = width * height
    if xy_area <= 0:
        return 0.0
    return cloud.size() / xy_area


def _delete_cloud_list(clouds: list, label: str) -> None:
    """Explicitly delete every cloud in *clouds* to free C++ memory."""
    for c in clouds:
        try:
            cc.deleteEntity(c)
        except Exception as exc:
            print(f"[COMP] WARNING — failed to delete entity ({label}): {exc}")
    clouds.clear()


def run_connected_component_analysis(
    c2017,
    c2018,
    c2020,
    c2c_1718: np.ndarray,
    c2c_1820: np.ndarray,
    grid_id: str,
    config: dict,
    out_dirs: dict[str, str | Path],
) -> dict[str, Any]:
    """Full connected-component gap analysis for one grid tile.

    This is the main entry point called by the pipeline after C2C arrays
    have been extracted.

    Parameters
    ----------
    c2017, c2018, c2020 : ccPointCloud
        Epoch clouds (already subsampled, globally shifted).
    c2c_1718 : np.ndarray
        C2C distances 2017 → 2018 (hurricane damage), float64.
    c2c_1820 : np.ndarray
        C2C distances 2018 → 2020 (recovery), float64.
    grid_id : str
        Tile / grid identifier.
    config : dict
        Configuration (thresholds, minimum sizes, etc.).
    out_dirs : dict
        Must contain at least ``"plots"`` key pointing to the output
        directory for figures.

    Returns
    -------
    results : dict
        Aggregated statistics for CSV output, keyed by epoch.
    """
    plots_dir = Path(out_dirs.get("plots", "outputs/plots"))

    # --- Step 1: overlap verification ---
    clouds_dict = {"2017": c2017, "2018": c2018, "2020": c2020}
    try:
        clipped, overlap_report = verify_and_clip_overlap(
            clouds_dict,
            grid_id,
            buffer_m=_cfg(config, "overlap_buffer_m"),
            config=config,
        )
    except ValueError:
        return {
            "grid_id": grid_id,
            "status": "skipped_no_overlap",
            "overlap_report": None,
        }

    c2017 = clipped["2017"]
    c2018 = clipped["2018"]
    c2020 = clipped["2020"]

    # --- Step 2: point density estimate from 2017 cloud ---
    point_density = _estimate_point_density(c2017)
    print(
        f"[COMP] {grid_id}: estimated point density = "
        f"{point_density:.1f} pts/m^2"
    )

    results: dict[str, Any] = {
        "grid_id": grid_id,
        "status": "ok",
        "overlap_report": overlap_report,
        "point_density_per_m2": point_density,
    }

    # --- Step 3: Damage epoch (2017 → 2018) ---
    components_17, residuals_17, oct_17 = extract_damage_components(
        c2017, c2c_1718, "C2C_2017_2018", grid_id, "damage_1718", config,
    )

    stats_1718, resid_summary_17 = compute_component_stats(
        components_17, residuals_17, grid_id, "damage_1718", point_density,
    )

    fit_17 = gap_size_distribution_analysis(
        stats_1718, grid_id, "damage_1718", plots_dir,
    )

    # Free component memory immediately
    _delete_cloud_list(components_17, "damage_1718 components")
    _delete_cloud_list(residuals_17, "damage_1718 residuals")

    results["damage_1718"] = {
        "component_stats": stats_1718,
        "residual_summary": resid_summary_17,
        "power_law_fit": {
            k: v for k, v in fit_17.items() if k != "areas_fitted"
        },
        "octree_level": oct_17,
    }

    # --- Step 4: Recovery epoch (2018 → 2020) ---
    components_18, residuals_18, oct_18 = extract_damage_components(
        c2018, c2c_1820, "C2C_2018_2020", grid_id, "recovery_1820", config,
    )

    stats_1820, resid_summary_18 = compute_component_stats(
        components_18, residuals_18, grid_id, "recovery_1820", point_density,
    )

    fit_18 = gap_size_distribution_analysis(
        stats_1820, grid_id, "recovery_1820", plots_dir,
    )

    _delete_cloud_list(components_18, "recovery_1820 components")
    _delete_cloud_list(residuals_18, "recovery_1820 residuals")

    results["recovery_1820"] = {
        "component_stats": stats_1820,
        "residual_summary": resid_summary_18,
        "power_law_fit": {
            k: v for k, v in fit_18.items() if k != "areas_fitted"
        },
        "octree_level": oct_18,
    }

    # --- Step 5: Epoch comparison ---
    epoch_comparison = compare_epochs(
        stats_1718, stats_1820, grid_id, plots_dir,
    )
    results["epoch_comparison"] = epoch_comparison

    print(f"[COMP] {grid_id}: connected-component analysis complete.")
    return results
