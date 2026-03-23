# LiDAR Gap Analysis — Project Context

## Project Overview
This project performs gap analysis on LiDAR point cloud data using cloudComPy (Python bindings for CloudCompare). The workflow is driven by a markdown input file (`data/input.md`) that specifies processing parameters and file paths.

Remote execution is handled via RealVNC cloud computing.

## Stack
- **Language**: Python 3.10+
- **Core library**: cloudComPy (CloudCompare Python bindings)
- **Remote access**: RealVNC
- **Input format**: Markdown files with structured parameters
- **Output**: Processed point clouds, gap metrics, and reports in `outputs/`

## Project Structure
```
lidar-gap-analysis/
├── src/
│   ├── parse_input.py     # Reads and validates data/input.md
│   ├── process.py         # Main cloudComPy processing pipeline
│   └── report.py          # Output formatting and reporting
├── data/
│   └── input.md           # Processing parameters and file paths
├── outputs/               # Generated files (gitignored)
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

## Workflow
1. Edit `data/input.md` with scan file paths and processing parameters
2. Run `src/process.py` — it reads `data/input.md` and executes the cloudComPy pipeline
3. Results are written to `outputs/`

## Conventions
- All file paths in `data/input.md` are relative to the project root
- cloudComPy must be installed and CloudCompare available on PATH
- Do not commit raw `.las`/`.laz` files — they go in `outputs/` which is gitignored
- Keep processing logic in `src/process.py`, parsing logic in `src/parse_input.py`

## Notes
- cloudComPy installation is environment-specific — see requirements.txt for details
- When running via RealVNC, ensure the remote environment has CloudCompare installed
