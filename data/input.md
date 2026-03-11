# LiDAR Processing Input

## Scan Files
<!-- List paths to your .las / .laz input files, one per line -->
- data/scans/scan_01.las

## Parameters

| Parameter         | Value  | Notes                          |
|------------------|--------|--------------------------------|
| voxel_size       | 0.05   | Metres, used for subsampling   |
| gap_threshold    | 0.5    | Min gap height in metres       |
| min_gap_area     | 1.0    | Min gap area in m²             |
| octree_level     | 8      | CloudCompare octree depth      |

## Output
- outputs/gap_report.txt
- outputs/gap_cloud.las

## Notes
<!-- Add any run-specific notes here -->
