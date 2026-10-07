#!/usr/bin/env python3
"""Inspect local safetensors metadata on CPU; no model load or GPU allocation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from safetensors import safe_open


DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64,
}
DENSE_FLOATS = {"BF16", "F16", "F32", "F64"}


def scan(model_dir: Path) -> dict:
    model_dir = model_dir.resolve(strict=True)
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The weight index must contain a nonempty weight_map")
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = [p.name for p in sorted(model_dir.glob("*.safetensors"))]
    if not shard_names:
        raise ValueError("No safetensors weights found")

    tensors = []
    seen = set()
    for shard_name in shard_names:
        shard = (model_dir / shard_name).resolve(strict=True)
        if not shard.is_relative_to(model_dir):
            raise ValueError(f"Shard is outside model directory: {shard_name}")
        # get_slice exposes shape/dtype without materializing any weight tensor.
        with safe_open(shard, framework="numpy", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"Duplicate tensor: {name}")
                if weight_map is not None and weight_map.get(name) != shard_name:
                    raise ValueError(f"Index/shard mismatch: {name}")
                seen.add(name)
                metadata = handle.get_slice(name)
                shape, dtype = metadata.get_shape(), metadata.get_dtype()
                if dtype not in DTYPE_BITS:
                    raise ValueError(f"Unsupported storage dtype {dtype}: {name}")
                elements = math.prod(shape)
                tensors.append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "stored_elements": elements,
                    "bytes": elements * DTYPE_BITS[dtype] // 8,
                })
    if weight_map is not None and set(weight_map) != seen:
        raise ValueError("Index references tensors missing from the shards")
    if not tensors:
        raise ValueError("Checkpoint contains no tensors")

    total_bytes = sum(t["bytes"] for t in tensors)
    groups = {}
    for tensor in tensors:
        # Group repeated numeric path components without assuming a model family.
        pattern = re.sub(r"(?<=\.)\d+(?=\.|$)", "*", tensor["name"])
        key = (pattern, tensor["dtype"], tuple(tensor["shape"]))
        if key not in groups:
            groups[key] = {
                "pattern": pattern, "dtype": tensor["dtype"],
                "shape": tensor["shape"], "count": 0,
                "stored_elements": 0, "bytes": 0,
            }
        row = groups[key]
        row["count"] += 1
        row["stored_elements"] += tensor["stored_elements"]
        row["bytes"] += tensor["bytes"]

    rows = sorted(groups.values(), key=lambda r: (-r["bytes"], r["pattern"]))
    for row in rows:
        row["percent"] = row["bytes"] / total_bytes * 100 if total_bytes else 0
        # Only dense floating-point *.weight matrices get hypothetical estimates.
        eligible = (row["pattern"].endswith(".weight")
                    and row["dtype"] in DENSE_FLOATS
                    and len(row["shape"]) == 2 and min(row["shape"]) > 0)
        row["ideal_saved_bytes"] = None
        row["dimensions_divisible_by_128"] = None
        if eligible:
            elements = math.prod(row["shape"])
            row["ideal_saved_bytes"] = {
                str(bits): row["bytes"] - row["count"] * ((elements * bits + 7) // 8)
                for bits in (8, 4)
            }
            row["dimensions_divisible_by_128"] = all(n % 128 == 0 for n in row["shape"])
    return {
        "model_dir": str(model_dir), "shard_count": len(shard_names),
        "tensor_count": len(tensors), "tensor_bytes": total_bytes,
        "notes": [
            "Stored tensor inventory, not deduplicated logical model parameters.",
            "Bytes exclude file headers, config, tokenizer and other auxiliary files.",
            "8/4-bit savings are hypothetical dense weight payload savings only; "
            "exclude scales, zero points, padding and retained higher-precision weights.",
            "128 divisibility is a shape check, not a kernel compatibility verdict.",
            "No PPL, inference speed or runtime memory is measured.",
        ],
        "groups": rows, "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--json", type=Path, help="Also save complete metadata as JSON")
    args = parser.parse_args()
    try:
        report = scan(args.model_dir)
        if args.json:
            # Do not modify the scanned model, even when invoked outside its read-only mount.
            if args.json.resolve().is_relative_to(Path(report["model_dir"])):
                raise ValueError("Write the report outside the input model directory")
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        parser.exit(1, f"Inventory failed: {exc}\n")
    print(f"{report['shard_count']} shards; {report['tensor_count']} tensors; "
          f"{report['tensor_bytes'] / 1024**3:.3f} GiB tensor payload")
    print("Pattern | dtype | shape | count | elements | MiB | % | "
          "ideal saved MiB (8b/4b) | dims divisible by 128")
    for row in report["groups"]:
        saved = row["ideal_saved_bytes"]
        savings = (f"{saved['8'] / 1024**2:.2f}/{saved['4'] / 1024**2:.2f}"
                   if saved is not None else "n/a")
        divisible = row["dimensions_divisible_by_128"]
        check = "n/a" if divisible is None else ("yes" if divisible else "no")
        print(f"{row['pattern']} | {row['dtype']} | {row['shape']} | "
              f"{row['count']} | {row['stored_elements']} | "
              f"{row['bytes'] / 1024**2:.2f} | {row['percent']:.4f} | {savings} | {check}")
    for note in report["notes"]:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
