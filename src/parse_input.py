"""Reads data/input.md and returns structured processing parameters."""

import re
from pathlib import Path


def parse_input(md_path: str = "data/input.md") -> dict:
    text = Path(md_path).read_text()

    scan_files = re.findall(r"^- (.+\.la[sz])\s*$", text, re.MULTILINE)

    params = {}
    for row in re.finditer(r"\|\s*([\w_]+)\s*\|\s*([\d.]+)\s*\|", text):
        params[row.group(1)] = float(row.group(2))

    outputs = re.findall(r"^- (outputs/.+)\s*$", text, re.MULTILINE)

    return {
        "scan_files": scan_files,
        "params": params,
        "outputs": outputs,
    }


if __name__ == "__main__":
    import json
    result = parse_input()
    print(json.dumps(result, indent=2))
