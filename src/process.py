"""Main LiDAR gap analysis pipeline using cloudComPy."""

import cloudComPy as cc
from parse_input import parse_input


def run_pipeline(config: dict):
    params = config["params"]

    for scan_path in config["scan_files"]:
        print(f"Loading: {scan_path}")
        cloud = cc.loadPointCloud(scan_path)
        if cloud is None:
            print(f"  [WARN] Failed to load {scan_path}, skipping.")
            continue

        # Subsample
        cloud = cc.CloudSamplingTools.resampleCloudWithOctreeAtLevel(
            cloud, int(params.get("octree_level", 8))
        )

        # TODO: implement gap detection logic here
        print(f"  Points after subsampling: {cloud.size()}")

    print("Pipeline complete.")


if __name__ == "__main__":
    config = parse_input()
    run_pipeline(config)
