#!/usr/bin/env python3
"""Run a small LeWM Cartpole MPC parameter sweep and rank the results.

This wrapper intentionally keeps the candidate list compact.  IsaacLab startup
is expensive, so the sweep focuses around the best-performing region found so
far: latent target + continuous action prior + hard edge rescue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SCRIPT = Path(__file__).resolve().with_name("isaaclab_lewm_mpc_cartpole.py")
DEFAULT_OUT_DIR = Path("/home/hall/code/.stable-wm/eval/mpc_sweep")


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any]


COMMON_PARAMS: dict[str, Any] = {
    "objective": "latent-target",
    "horizon": 4,
    "num-candidates": 2048,
    "cem-iters": 4,
    "force-rescue-candidates": True,
    "action-prior-weight": 1.2,
    "prior-pole-kp": 2.4,
    "prior-pole-kd": 0.6,
    "prior-cart-kp": 0.08,
    "prior-cart-kd": 0.02,
    "edge-rescue-weight": 8.0,
    "edge-rescue-threshold": 2.3,
    "edge-rescue-velocity-threshold": 0.2,
    "edge-rescue-return-action": 1.0,
    "edge-rescue-prior-suppression": 1.0,
    "edge-rescue-gate-scale": 0.15,
    "save-step-diagnostics": True,
}


CANDIDATES: list[Candidate] = [
    Candidate("edge_gate015_baseline", {}),
    Candidate("pole30_kd08", {"prior-pole-kp": 3.0, "prior-pole-kd": 0.8}),
    Candidate(
        "pole32_kd09_weight14",
        {"action-prior-weight": 1.4, "prior-pole-kp": 3.2, "prior-pole-kd": 0.9},
    ),
    Candidate(
        "edge_early20_pole30",
        {"prior-pole-kp": 3.0, "prior-pole-kd": 0.8, "edge-rescue-threshold": 2.0},
    ),
    Candidate(
        "edge_early21_gate012",
        {
            "prior-pole-kp": 3.0,
            "prior-pole-kd": 0.8,
            "edge-rescue-threshold": 2.1,
            "edge-rescue-gate-scale": 0.12,
        },
    ),
    Candidate(
        "edge_weight12_gate012",
        {
            "prior-pole-kp": 3.0,
            "prior-pole-kd": 0.8,
            "edge-rescue-weight": 12.0,
            "edge-rescue-gate-scale": 0.12,
        },
    ),
    Candidate(
        "direction_edge_soft",
        {
            "action-prior-weight": 0.0,
            "direction-bias-weight": 0.28,
            "direction-pole-weight": 1.5,
            "direction-pole-vel-weight": 0.4,
            "edge-rescue-weight": 8.0,
            "edge-rescue-threshold": 2.2,
            "edge-rescue-gate-scale": 0.15,
        },
    ),
    Candidate(
        "direction_edge_cart",
        {
            "action-prior-weight": 0.0,
            "direction-bias-weight": 0.25,
            "direction-pole-weight": 1.5,
            "direction-pole-vel-weight": 0.35,
            "direction-cart-weight": 0.08,
            "direction-cart-vel-weight": 0.02,
            "edge-rescue-weight": 8.0,
            "edge-rescue-threshold": 2.2,
            "edge-rescue-gate-scale": 0.15,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep LeWM Cartpole MPC parameters.")
    parser.add_argument("--python", default=sys.executable, help="Python executable from the IsaacLab environment.")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-len", type=int, default=300)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--save-gif-best", action="store_true", help="After sweep, rerun the best candidate with GIF enabled.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing candidate JSON files instead of rerunning them.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def flag_name(name: str) -> str:
    return "--" + name


def build_command(args: argparse.Namespace, candidate: Candidate, out_path: Path, save_gif: bool = False) -> list[str]:
    params = dict(COMMON_PARAMS)
    params.update(candidate.params)

    cmd = [
        args.python,
        str(args.script),
        "--device",
        args.device,
        "--episodes",
        str(args.episodes),
        "--episode-len",
        str(args.episode_len),
        "--out",
        str(out_path),
    ]
    if args.headless:
        cmd.append("--headless")

    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(flag_name(key))
        else:
            cmd.extend([flag_name(key), str(value)])

    if save_gif:
        cmd.extend(
            [
                "--save-gif",
                "--gif-out",
                str(out_path.with_suffix(".gif")),
            ]
        )
    return cmd


def score_result(result: dict[str, Any]) -> float:
    summary = result["summary"]
    episodes = result["episodes"]
    mean_survival = float(summary["mean_survival_steps"])
    mean_angle = float(summary["mean_abs_pole_angle"])
    mean_cart = sum(float(ep["mean_abs_cart_pos"]) for ep in episodes) / max(1, len(episodes))
    terminated = float(summary["terminated_episodes"])

    # Survival dominates.  Cart centring is the next priority, then pole angle.
    return mean_survival - 35.0 * mean_cart - 80.0 * mean_angle - 20.0 * terminated


def summarize_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    episodes = result["episodes"]
    mean_cart = sum(float(ep["mean_abs_cart_pos"]) for ep in episodes) / max(1, len(episodes))
    edge_align = None
    active_edges = []
    for ep in episodes:
        for step in ep.get("steps", []):
            edge = step.get("edge_rescue")
            if edge and edge.get("active"):
                active_edges.append(1.0 if step["action"][0] * edge["target_action"] > 0 else 0.0)
    if active_edges:
        edge_align = sum(active_edges) / len(active_edges)

    return {
        "path": str(path),
        "score": score_result(result),
        "mean_survival": result["summary"]["mean_survival_steps"],
        "mean_angle": result["summary"]["mean_abs_pole_angle"],
        "mean_cart": mean_cart,
        "terminated": result["summary"]["terminated_episodes"],
        "edge_align": edge_align,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = CANDIDATES[args.start_index :]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    summaries = []
    for index, candidate in enumerate(candidates, start=args.start_index):
        out_path = args.out_dir / f"{index:02d}_{candidate.name}.json"
        cmd = build_command(args, candidate, out_path)
        print("\n[SWEEP]", index, candidate.name)
        print(" ".join(cmd))
        if not args.dry_run:
            if args.skip_existing and out_path.exists():
                print(f"[SWEEP] reuse existing {out_path}")
            else:
                subprocess.run(cmd, check=True)
            summaries.append({"name": candidate.name, **summarize_result(out_path)})

    if args.dry_run:
        return

    summaries.sort(key=lambda item: item["score"], reverse=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print("\n[SWEEP] ranking")
    for rank, item in enumerate(summaries, start=1):
        print(
            f"{rank:02d}. {item['name']} score={item['score']:.3f} "
            f"survival={item['mean_survival']:.1f} angle={item['mean_angle']:.4f} "
            f"cart={item['mean_cart']:.4f} terminated={item['terminated']} "
            f"edge_align={item['edge_align']}"
        )
    print(f"[SWEEP] wrote {summary_path}")

    if args.save_gif_best and summaries:
        best = next(candidate for candidate in CANDIDATES if candidate.name == summaries[0]["name"])
        out_path = args.out_dir / f"best_{best.name}.json"
        cmd = build_command(args, best, out_path, save_gif=True)
        print("\n[SWEEP] rerun best with GIF")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
