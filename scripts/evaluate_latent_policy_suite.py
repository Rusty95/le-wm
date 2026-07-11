#!/usr/bin/env python3
"""Run a small IsaacLab evaluation suite for the LeWM latent policy.

This script is intentionally a subprocess orchestrator: each rollout launches a
fresh IsaacLab process through ``isaaclab_lewm_policy_cartpole.py``.  That keeps
Isaac Sim/AppLauncher lifecycle issues out of the benchmark loop and produces a
single JSON plus a Markdown table for reporting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = Path("/home/hall/code/.stable-wm/eval/latent_policy_suite")
DEFAULT_CHECKPOINT = Path("/home/hall/code/.stable-wm/checkpoints/lewm_full_angle_multistep_h10/weights_epoch_100.pt")
DEFAULT_POLICY_HEAD = Path("/home/hall/code/.stable-wm/checkpoints/lewm_full_angle_latent_policy.pt")
DEFAULT_ACTION_STATS = Path("/home/hall/code/.stable-wm/datasets/isaaclab_full_angle_120k.h5")


@dataclass(frozen=True)
class Scenario:
    name: str
    episode_len: int
    episode_length_s: float
    angle_range: tuple[float, float]
    disturbance: bool = False


SCENARIOS = {
    "near_upright": Scenario(
        name="near_upright",
        episode_len=300,
        episode_length_s=25.0,
        angle_range=(-0.25, 0.25),
    ),
    "bottom": Scenario(
        name="bottom",
        episode_len=300,
        episode_length_s=25.0,
        angle_range=(2.8, 3.14),
    ),
    "disturbance": Scenario(
        name="disturbance",
        episode_len=1200,
        episode_length_s=100.0,
        angle_range=(-0.25, 0.25),
        disturbance=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate the LeWM latent-policy Cartpole controller.")
    parser.add_argument("--task", default="RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--policy-head", type=Path, default=DEFAULT_POLICY_HEAD)
    parser.add_argument("--action-stats-h5", type=Path, default=DEFAULT_ACTION_STATS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9317, 9323, 9331])
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["near_upright", "bottom", "disturbance"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--save-gif", action="store_true", help="Save one GIF per rollout. This is slower and uses more disk.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_disturbance_args(cmd: list[str]) -> None:
    cmd.extend(
        [
            "--disturbance-start-step",
            "60",
            "--disturbance-interval",
            "160",
            "--disturbance-count",
            "5",
            "--disturbance-min",
            "2.4",
            "--disturbance-max",
            "6.0",
            "--disturbance-stable-steps",
            "60",
            "--disturbance-angle-threshold",
            "0.15",
            "--disturbance-pole-vel-threshold",
            "0.8",
            "--disturbance-cart-threshold",
            "0.8",
            "--disturbance-cart-vel-threshold",
            "0.5",
        ]
    )


def build_command(args: argparse.Namespace, scenario: Scenario, seed: int, out_json: Path, gif_path: Path) -> list[str]:
    script = REPO_DIR / "scripts" / "isaaclab_lewm_policy_cartpole.py"
    cmd = [
        args.python,
        str(script),
        "--task",
        args.task,
        "--checkpoint",
        str(args.checkpoint),
        "--policy-head",
        str(args.policy_head),
        "--action-stats-h5",
        str(args.action_stats_h5),
        "--episodes",
        "1",
        "--episode-len",
        str(scenario.episode_len),
        "--episode-length-s",
        str(scenario.episode_length_s),
        "--initial-pole-angle-range",
        str(scenario.angle_range[0]),
        str(scenario.angle_range[1]),
        "--high-contrast-scene",
        "--out",
        str(out_json),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if scenario.disturbance:
        add_disturbance_args(cmd)
    if args.save_gif:
        cmd.extend(["--save-gif", "--gif-out", str(gif_path)])
    return cmd


def read_episode(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    return {
        "reward_sum": float(episode["reward_sum"]),
        "survival_steps": int(episode["survival_steps"]),
        "done_count": int(episode["done_count"]),
        "mean_abs_pole_angle": float(episode["mean_abs_pole_angle"]),
        "max_abs_pole_angle": float(episode["max_abs_pole_angle"]),
        "mean_abs_cart_pos": float(episode["mean_abs_cart_pos"]),
        "mean_abs_action": float(episode["mean_abs_action"]),
        "disturbance_count": len(episode.get("disturbances", [])),
        "recovered_disturbances": sum(1 for item in episode.get("disturbances", []) if item.get("recovery_step") is not None),
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for scenario in sorted({row["scenario"] for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        summary = {"scenario": scenario, "runs": len(subset)}
        for key in (
            "reward_sum",
            "survival_steps",
            "done_count",
            "mean_abs_pole_angle",
            "max_abs_pole_angle",
            "mean_abs_cart_pos",
            "mean_abs_action",
            "recovered_disturbances",
        ):
            values = [float(row[key]) for row in subset]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def markdown_table(summaries: list[dict]) -> str:
    lines = [
        "# LeWM Latent Policy Evaluation Suite",
        "",
        "| Scenario | Runs | Survival | Reward | Mean angle | Max angle | Cart | Action | Recovered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {scenario} | {runs} | {surv:.1f}±{surv_std:.1f} | {reward:.1f}±{reward_std:.1f} | "
            "{angle:.3f}±{angle_std:.3f} | {max_angle:.3f}±{max_angle_std:.3f} | "
            "{cart:.3f}±{cart_std:.3f} | {action:.3f}±{action_std:.3f} | {rec:.1f}±{rec_std:.1f} |".format(
                scenario=item["scenario"],
                runs=item["runs"],
                surv=item["survival_steps_mean"],
                surv_std=item["survival_steps_std"],
                reward=item["reward_sum_mean"],
                reward_std=item["reward_sum_std"],
                angle=item["mean_abs_pole_angle_mean"],
                angle_std=item["mean_abs_pole_angle_std"],
                max_angle=item["max_abs_pole_angle_mean"],
                max_angle_std=item["max_abs_pole_angle_std"],
                cart=item["mean_abs_cart_pos_mean"],
                cart_std=item["mean_abs_cart_pos_std"],
                action=item["mean_abs_action_mean"],
                action_std=item["mean_abs_action_std"],
                rec=item["recovered_disturbances_mean"],
                rec_std=item["recovered_disturbances_std"],
            )
        )
    lines.extend(
        [
            "",
            "Metrics are averaged over seeds. Angle and cart values use absolute values.",
            "The `Recovered` column counts disturbance events that returned to the configured stable band.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.summary_json or args.out_dir / "summary.json"
    summary_md = args.summary_md or args.out_dir / "summary.md"
    gif_dir = args.out_dir / "gifs"
    if args.save_gif:
        gif_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        for seed in args.seeds:
            out_json = args.out_dir / f"{scenario.name}_seed{seed}.json"
            gif_path = gif_dir / f"{scenario.name}_seed{seed}.gif"
            cmd = build_command(args, scenario, seed, out_json, gif_path)
            print("$ " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True, timeout=args.timeout_s)
            row = {
                "scenario": scenario.name,
                "seed": seed,
                "json": str(out_json),
                **read_episode(out_json),
            }
            if args.save_gif:
                row["gif"] = str(gif_path)
            rows.append(row)

    if args.dry_run:
        return

    summaries = summarize(rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "policy_head": str(args.policy_head),
        "action_stats_h5": str(args.action_stats_h5),
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "rows": rows,
        "summary": summaries,
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(markdown_table(summaries), encoding="utf-8")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
