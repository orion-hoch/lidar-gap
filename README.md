# LiDAR Gap Analysis — Puerto Rico G-LiHT

Quantifies hurricane-driven canopy gap formation in the **Luquillo Experimental Forest** using repeat airborne LiDAR surveys:

- **Pre-Maria**: March 2017 (G-LiHT, NASA)
- **Post-Maria**: May 2018 (G-LiHT, NASA)

---

## Overview

The pipeline computes per-point cloud-to-cloud (C2C) distances between the two epochs, rasterises the result onto a 2D grid, and delineates canopy gaps as spatially contiguous regions of high C2C distance. The goal throughout is to **preserve as much original point-level information as possible** — no permanent subsampling is applied to the source data.

---

## Data Pathway

### Step 1 — Load & Merge (`load_and_merge_folder`)

Every `.las` / `.las.gz` tile in each epoch folder is loaded sequentially. A per-file timeout (default 120 s) skips tiles that hang. Files are merged into two single clouds using `cc.MergeEntities`.

- `SUBSAMPLE_N_PER_FILE = None` — subsampling is **disabled**; all original points are retained.
- A global coordinate shift is read from a probe file and applied uniformly to every tile so that all files share the same local reference frame and can be merged without precision loss.

**Output:** two in-memory merged clouds — `merged_2017`, `merged_2018`.

---

### Step 2 — Overlap Detection & Clipping (`clip_to_actual_overlap`)

The two merged clouds are not guaranteed to cover exactly the same area (irregular flight strip edges). To avoid computing C2C across non-overlapping regions, the pipeline:

1. Bins each cloud into a 2D voxel grid at `OVERLAP_VOXEL_M` (50 m) resolution.
2. Takes the **intersection** of occupied voxel cells to find the true shared footprint.
3. Expands the shared bounding box by `OVERLAP_BUFFER_M` (100 m) on all sides.
4. Clips both clouds to this extended bounding box using a scalar-field filter.

Points outside the shared footprint are dropped here — this is the only permanent point reduction in the pipeline, and is unavoidable since C2C is undefined where only one epoch exists. The original `.las` files on disk are not modified.

**Output:** `c2017_clipped`, `c2018_clipped` — both confined to the overlap region.

---

### Step 3 — Cloud-to-Cloud Distance (`compute_c2c`)

Computes the nearest-neighbour distance from every point in `c2017_clipped` to the 2018 cloud using CloudCompare's multithreaded C2C engine. The result is attached to `c2017_clipped` as a scalar field (`C2C_2017_2018`).

No points are removed. The 2018 cloud is freed after this step.

The cloud with the C2C scalar field is saved to disk as `BIN_DIR/fullarea_2017_c2c.bin` — this is the canonical intermediate file used by all subsequent analysis steps.

**Output:** `fullarea_2017_c2c.bin` — full 2017 point cloud with per-point C2C distances.

---

### Step 4 — Connected Components, Bbox Method (`extract_damage_components`)

An earlier gap delineation approach kept for comparison. The cloud is filtered to points with C2C > `damage_threshold` (2 m), and `cc.ExtractConnectedComponents` is run on those points using an octree targeted at ~2 m cell size.

**Limitation:** gap boundaries are defined by where *damaged* points end, not where intact canopy resumes, so areas are systematically underestimated. This method is retained for the power-law comparison figures only.

---

### Step 5 — Statistical Analysis & Power-Law Fit

Gap areas from Step 4 are fitted with a power-law distribution using both MLE and OLS log-log regression. A log-normal alternative is also fitted and the preferred model identified by log-likelihood comparison.

**Outputs (CSV):** `c2c_summary.csv`, `c2c_thresholds.csv`, `c2c_extreme.csv`, `component_stats.csv`, `power_law_fit.csv`.

---

### Step 6 — Raster-Based Gap Delineation (primary method)

Reloads `fullarea_2017_c2c.bin` and performs gap delineation on the **complete unfiltered point cloud**:

1. **Rasterise** — bin all points into a 2D grid at `RASTER_RES_M` (2 m) resolution. Each cell stores the **mean C2C** of all points that fall in it. Cells with no points are left as `NaN`. C2C values are clipped to `NOISE_CEIL_M` (20 m) in the numpy array only before averaging; the saved cloud is unaffected.

2. **Damage mask** — threshold the raster: cells where `mean_c2c > DAMAGE_THRESH` (2 m) and that contain at least one point become `True`.

3. **Connected components** — `scipy.ndimage.label` with 8-connectivity labels each contiguous blob of damaged cells as a distinct gap.

4. **Measure** — for each component: pixel count → area (pixels × 4 m²), mean C2C, max C2C.

5. **Filter** — discard components smaller than `min_gap_area_m2` (10 m²) or with mean C2C above `NOISE_CEIL_M` (20 m).

6. **Power-law fit** — same MLE/OLS procedure as Step 5, applied to raster-derived areas.

Because intact-canopy points (C2C < 2 m) are included in the raster, they form the true gap boundary. This is the primary quantitative result.

**Outputs (CSV):** `raster_component_stats.csv`, `raster_power_law_fit.csv`.

---

### Step 7 — LAS Export for Visual Inspection

Reloads `fullarea_2017_c2c.bin`, maps every original point to its raster cell, and looks up the component label for that cell. Points that fall inside a valid gap component are kept; all others are discarded.

The per-point component ID is written as a scalar field (`component_id`) alongside the original C2C value (`c2c_m`). The global coordinate shift is restored so the file aligns correctly with source data in CloudCompare or any other LAS viewer.

**Output:** `BIN_DIR/raster_components.las` — original-density point cloud, gap points only, colourable by `component_id` or `c2c_m`.

---

## Outputs

| File | Location | Description |
|---|---|---|
| `fullarea_2017_c2c.bin` | `BIN_DIR` | Full 2017 cloud + C2C scalar field |
| `raster_components.las` | `BIN_DIR` | Gap points at original density, tagged with component ID |
| `c2c_summary.csv` | `CSV_DIR` | C2C distribution statistics |
| `c2c_thresholds.csv` | `CSV_DIR` | Fraction of points above each threshold |
| `c2c_extreme.csv` | `CSV_DIR` | Extreme value report (>25 m) |
| `component_stats.csv` | `CSV_DIR` | Per-component stats, bbox method |
| `raster_component_stats.csv` | `CSV_DIR` | Per-component stats, raster method |
| `power_law_fit.csv` | `CSV_DIR` | Power-law fit, bbox method |
| `raster_power_law_fit.csv` | `CSV_DIR` | Power-law fit, raster method |
| `fullarea_c2c_analysis.png` | `PLOT_DIR` | C2C histogram and CDF |
| `fullarea_gap_distribution.png` | `PLOT_DIR` | Gap size distribution |
| `gap_delineation_comparison.png` | `PLOT_DIR` | Bbox vs raster method comparison |

---

## Configuration

Key parameters in the configuration cell:

| Parameter | Default | Description |
|---|---|---|
| `SUBSAMPLE_N_PER_FILE` | `None` | Per-file point cap; `None` = load all points |
| `FILE_LOAD_TIMEOUT_S` | `120` | Seconds before a hung tile is skipped |
| `THREADS` | `12` | C2C computation threads |
| `NOISE_CEIL_M` | `20.0` | C2C values above this treated as noise |
| `EXTREME_THRESHOLD_M` | `25.0` | Threshold for extreme-value reporting |
| `OVERLAP_VOXEL_M` | `50.0` | Voxel size for overlap detection |
| `OVERLAP_BUFFER_M` | `100.0` | Buffer added around shared overlap bbox |
| `RASTER_RES_M` | `2.0` | Raster cell size for gap delineation |
| `CC_CONFIG['damage_threshold']` | `2.0` | Min C2C (m) to classify a point/cell as damaged |
| `CC_CONFIG['min_gap_area_m2']` | `10.0` | Minimum gap area to retain |

---

## Dependencies

- [CloudComPy](https://github.com/CloudCompare/CloudComPy) (`cloudcompy`)
- NumPy, SciPy, Matplotlib
- laspy (optional — only needed if modifying the LAS export cell)
