#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fmt(x):
    if x is None:
        return "-"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", default=None, help="Path to timing_*.jsonl")
    parser.add_argument("--dir", default=None, help="Run directory containing metrics/timing_*.jsonl")
    args = parser.parse_args()

    timing_path = args.timing
    if timing_path is None:
        if args.dir is None:
            raise SystemExit("Use --timing PATH or --dir RUN_DIR")
        metrics_dir = os.path.join(args.dir, "metrics")
        candidates = [
            os.path.join(metrics_dir, p)
            for p in os.listdir(metrics_dir)
            if p.startswith("timing_") and p.endswith(".jsonl")
        ]
        if not candidates:
            raise FileNotFoundError(f"No timing_*.jsonl under {metrics_dir}")
        timing_path = sorted(candidates)[-1]

    rows = read_jsonl(timing_path)
    print(f"Timing file: {timing_path}")
    print(f"Records: {len(rows)}")

    by_phase = defaultdict(list)
    for r in rows:
        by_phase[r.get("phase", "unknown")].append(r)

    phase_rows = []
    for phase, items in by_phase.items():
        wall = [float(x.get("wall_sec", 0.0)) for x in items]
        cpu = [float(x.get("cpu_sec", 0.0)) for x in items if x.get("cpu_sec") is not None]
        peak = [float(x.get("cuda_peak_alloc_mb", 0.0)) for x in items if x.get("cuda_peak_alloc_mb") is not None]
        phase_rows.append([
            phase,
            len(items),
            fmt(sum(wall)),
            fmt(sum(wall) / max(1, len(wall))),
            fmt(max(wall) if wall else 0.0),
            fmt(sum(cpu) if cpu else None),
            fmt(max(peak) if peak else None),
        ])
    phase_rows.sort(key=lambda x: float(x[2]) if x[2] != "-" else 0.0, reverse=True)

    print("\n== By phase ==")
    print_table(
        ["phase", "count", "wall_sum_s", "wall_mean_s", "wall_max_s", "cpu_sum_s", "max_cuda_alloc_mb"],
        phase_rows,
    )

    task_rows = []
    for r in rows:
        if r.get("phase") in ("task_total", "trainer_task_loop_total"):
            task_rows.append([
                r.get("phase"),
                r.get("task"),
                r.get("total_classes"),
                fmt(r.get("wall_sec")),
                fmt(r.get("cpu_sec")),
                fmt(r.get("cuda_peak_alloc_mb")),
                fmt(r.get("top1")),
            ])
    if task_rows:
        task_rows.sort(key=lambda x: (str(x[0]), int(x[1])))
        print("\n== Task totals ==")
        print_table(
            ["phase", "task", "total_cls", "wall_s", "cpu_s", "cuda_peak_mb", "top1"],
            task_rows,
        )

    gen_detail_phases = {
        "generator_init",
        "generator_data_setup",
        "class_inversion_total",
        "synthesis_imgs_total",
    }
    gen_wall = sum(float(r.get("wall_sec", 0.0)) for r in rows if r.get("phase") in gen_detail_phases)
    all_task_wall = sum(float(r.get("wall_sec", 0.0)) for r in rows if r.get("phase") == "task_total")
    if all_task_wall > 0:
        print(f"\nGeneration-related logged wall time: {gen_wall:.2f}s ({gen_wall / all_task_wall * 100.0:.2f}% of task_total sum)")


if __name__ == "__main__":
    main()
