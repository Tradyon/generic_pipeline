#!/usr/bin/env python3
"""
Upload product schema rows from a CSV into the product-schema service.

Expected CSV columns:
- product: human readable product name
- product_id: schema identifier (sent as schema_id)
- schema_json: JSON object containing attribute lists
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from urllib import error, request

DEFAULT_URL = (
    "http://internal-tradyoninternallb-1670312611.me-central-1."
    "elb.amazonaws.com:8091/v1/product-schema/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload product schema rows to the product-schema API."
    )
    parser.add_argument(
        "--csv",
        default="output/onion/product_schema_master.csv",
        help="Path to product schema CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("PRODUCT_SCHEMA_URL", DEFAULT_URL),
        help="Product schema API URL (default: env PRODUCT_SCHEMA_URL or built-in)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads without sending them",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop on first failed upload instead of continuing",
    )
    return parser.parse_args()


def load_payloads(csv_path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield (row_number, payload) for each CSV row."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"product", "product_id", "schema_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_list}")

        for idx, row in enumerate(reader, start=1):
            try:
                attributes = json.loads(row["schema_json"])
                if not isinstance(attributes, dict):
                    raise ValueError("schema_json must decode to a JSON object")
            except Exception as exc:
                raise ValueError(f"Row {idx}: invalid schema_json ({exc})") from exc

            payload = {
                "name": row["product"],
                "schema_id": row["product_id"],
                "attributes": attributes,
            }
            yield idx, payload


def post_payload(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    with request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read().decode("utf-8")


def main() -> int:
    args = parse_args()
    url = args.url
    csv_path = Path(args.csv)

    if not url:
        sys.stderr.write("Product schema URL is required.\n")
        return 1

    if not csv_path.exists():
        sys.stderr.write(f"CSV not found: {csv_path}\n")
        return 1

    errors = 0
    for idx, payload in load_payloads(csv_path):
        prefix = f"[row {idx}] {payload['schema_id']} - {payload['name']}"

        if args.dry_run:
            print(f"{prefix} (dry-run):")
            print(json.dumps(payload, indent=2))
            continue

        try:
            status, response = post_payload(url, payload, args.timeout)
            print(f"{prefix} -> HTTP {status}")
            if response.strip():
                print(response)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            print(f"{prefix} failed: HTTP {exc.code} {exc.reason}")
            if body:
                print(body)
            errors += 1
            if args.stop_on_fail:
                break
        except error.URLError as exc:
            print(f"{prefix} failed: {exc}")
            errors += 1
            if args.stop_on_fail:
                break

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
