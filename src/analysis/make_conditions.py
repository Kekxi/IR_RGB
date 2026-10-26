"""Create a val visibility review table; never changes object detection labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--review-sample", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed table: {args.output}")
    try:
        sample_text = args.review_sample.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        sample_text = args.review_sample.read_text(encoding="gb18030")
    reviewed = {row["id"]: row for row in csv.DictReader(sample_text.splitlines()) if row["split"] == "val"}
    rows = []
    with (args.dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        for pair in csv.DictReader(f):
            if pair["split"] != "val" or pair["same_image_size"] != "1":
                continue
            prior = reviewed.get(pair["id"], {})
            rows.append({"id": pair["id"], "event": pair["event"],
                         "visibility": prior.get("review_weather_visibility", ""),
                         "illumination": prior.get("review_illumination", ""),
                         "reviewer": "", "notes": prior.get("review_notes", "")})
    if len(rows) != 877:
        raise ValueError(f"Expected 877 eligible val pairs, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows, prefilled {sum(bool(r['visibility']) for r in rows)} visibility entries")


if __name__ == "__main__":
    main()
