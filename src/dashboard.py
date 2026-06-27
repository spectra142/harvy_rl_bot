#!/usr/bin/env python3
"""
Training metrics dashboard using Gradio.

Usage:
    python src/dashboard.py --metrics_dir ./checkpoints/logs
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import gradio as gr
import numpy as np


def load_metrics(metrics_file: str) -> List[Dict[str, Any]]:
    """Load training metrics from JSON file."""
    if not os.path.exists(metrics_file):
        return []
    with open(metrics_file, encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_results(benchmark_file: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file."""
    if not os.path.exists(benchmark_file):
        return []
    try:
        with open(benchmark_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def create_dashboard(metrics_dir: str, refresh_interval: int = 5):
    """Create and launch the Gradio dashboard."""
    metrics_file = os.path.join(metrics_dir, "metrics.json")
    curriculum_file = os.path.join(metrics_dir, "curriculum_state.yaml")
    benchmark_file = os.path.join(
        metrics_dir, "..", "benchmarks", "benchmark_results.json"
    )

    def get_metrics():
        """Get latest metrics data."""
        metrics = load_metrics(metrics_file)
        if not metrics:
            return {}, {}, {}, "No metrics available"

        latest = metrics[-1]

        # Training progress plot
        timesteps = [m["timesteps"] for m in metrics]
        rewards = [m.get("mean_reward", 0) for m in metrics]
        ep_lengths = [m.get("mean_ep_length", 0) for m in metrics]

        progress_data = {
            "Timesteps": timesteps,
            "Mean Reward": rewards,
            "Mean Episode Length": ep_lengths,
        }

        # Summary stats
        summary = {
            "Total Timesteps": latest.get("timesteps", 0),
            "Latest Mean Reward": f"{latest.get('mean_reward', 0):.2f}",
            "Latest Episode Length": f"{latest.get('mean_ep_length', 0):.0f}",
            "Training Time": f"{(latest.get('time', time.time()) - metrics[0].get('time', time.time())) / 3600:.1f}h",
        }

        return progress_data, summary, metrics, "Metrics loaded successfully"

    def get_benchmark_data():
        """Get benchmark results data."""
        results = load_benchmark_results(benchmark_file)
        if not results:
            return [], {}, "No benchmark results available"

        # Get the latest evaluation round (most recent entry per benchmark)
        benchmark_names = [
            "survive_10min",
            "gather_16_logs",
            "craft_pickaxe",
            "mine_16_stone",
            "kill_hostile",
            "build_shelter",
            "survive_first_night",
        ]

        latest_scores: Dict[str, Dict[str, Any]] = {}
        best_scores: Dict[str, float] = {name: 0.0 for name in benchmark_names}

        for entry in results:
            name = entry.get("benchmark_name", "")
            if name in benchmark_names:
                score = entry.get("score", 0)
                # Track latest
                ts = entry.get("timestamp", 0)
                if name not in latest_scores or ts > latest_scores[name].get(
                    "timestamp", 0
                ):
                    latest_scores[name] = entry
                # Track best
                if score > best_scores[name]:
                    best_scores[name] = score

        # Build comparison table data
        table_data = []
        for name in benchmark_names:
            if name in latest_scores:
                entry = latest_scores[name]
                descriptions = {
                    "survive_10min": "Survive 10 minutes",
                    "gather_16_logs": "Gather 16 logs",
                    "craft_pickaxe": "Craft pickaxe",
                    "mine_16_stone": "Mine 16 stone",
                    "kill_hostile": "Kill hostile mob",
                    "build_shelter": "Build shelter",
                    "survive_first_night": "Survive first night",
                }
                table_data.append(
                    {
                        "Benchmark": descriptions.get(name, name),
                        "Current Score": f"{entry.get('score', 0):.3f}",
                        "Best Score": f"{best_scores.get(name, 0):.3f}",
                        "Success": "Yes" if entry.get("success", False) else "No",
                        "Raw Value": f"{entry.get('raw_value', 0):.1f}",
                    }
                )

        # Calculate overall score
        overall = float(np.mean(list(best_scores.values()))) if best_scores else 0.0

        summary = {
            "Overall Score": f"{overall:.3f}",
            "Total Evaluations": (
                len(results) // len(benchmark_names) if benchmark_names else 0
            ),
            "Benchmarks Run": len(latest_scores),
        }

        # Build history for line plot (overall score over time)
        # Group entries by approximate timestamp buckets
        history_data = {"Timesteps": [], "Overall Score": []}
        eval_rounds: Dict[str, Dict[str, Any]] = {}
        for entry in results:
            name = entry.get("benchmark_name", "")
            ts = str(entry.get("timestamp", 0))
            if ts not in eval_rounds:
                eval_rounds[ts] = {}
            eval_rounds[ts][name] = entry

        for ts, benchmarks in sorted(eval_rounds.items(), key=lambda x: float(x[0])):
            scores = [b.get("score", 0) for b in benchmarks.values()]
            if scores:
                history_data["Timesteps"].append(
                    datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
                )
                history_data["Overall Score"].append(float(np.mean(scores)))

        # Build bar chart data
        bar_data = {"Benchmark": [], "Current": [], "Best": []}
        for name in benchmark_names:
            descriptions = {
                "survive_10min": "Survive 10m",
                "gather_16_logs": "Gather Logs",
                "craft_pickaxe": "Craft Pick",
                "mine_16_stone": "Mine Stone",
                "kill_hostile": "Kill Mob",
                "build_shelter": "Build",
                "survive_first_night": "1st Night",
            }
            bar_data["Benchmark"].append(descriptions.get(name, name))
            bar_data["Current"].append(latest_scores.get(name, {}).get("score", 0))
            bar_data["Best"].append(best_scores.get(name, 0))

        return table_data, summary, history_data, bar_data, "Benchmarks loaded"

    # Build UI
    with gr.Blocks(title="RL Minecraft Bot Dashboard") as dashboard:
        gr.Markdown("# RL Minecraft Bot Training Dashboard")
        gr.Markdown("Monitor training progress, evaluate skills, and manage the bot.")

        with gr.Tab("Training Progress"):
            with gr.Row():
                status_text = gr.Textbox(label="Status", value="Loading...")
                refresh_btn = gr.Button("Refresh Now")

            with gr.Row():
                summary_table = gr.JSON(label="Summary Statistics")

            with gr.Row():
                reward_plot = gr.LinePlot(
                    label="Mean Reward over Timesteps",
                    x="Timesteps",
                    y="Mean Reward",
                )

            with gr.Row():
                length_plot = gr.LinePlot(
                    label="Episode Length over Timesteps",
                    x="Timesteps",
                    y="Mean Episode Length",
                )

        with gr.Tab("Benchmarks"):
            gr.Markdown("## Skill Benchmarks")
            gr.Markdown(
                "Harvy is evaluated on 7 skill benchmarks that measure "
                "progressive mastery of Minecraft survival."
            )

            with gr.Row():
                bench_status = gr.Textbox(label="Status", value="Loading...")
                bench_refresh_btn = gr.Button("Refresh Benchmarks")

            with gr.Row():
                bench_summary = gr.JSON(label="Benchmark Summary")

            with gr.Row():
                bench_table = gr.Dataframe(
                    label="Benchmark Scores (Current vs Best)",
                    headers=[
                        "Benchmark",
                        "Current Score",
                        "Best Score",
                        "Success",
                        "Raw Value",
                    ],
                )

            with gr.Row():
                bench_overall_plot = gr.LinePlot(
                    label="Overall Benchmark Score over Time",
                    x="Timesteps",
                    y="Overall Score",
                )

            with gr.Row():
                bench_bar_plot = gr.BarPlot(
                    label="Benchmark Performance Comparison",
                    x="Benchmark",
                    y="Current",
                    color="Benchmark",
                )

        with gr.Tab("Skill Evaluation"):
            gr.Markdown("## Skill Performance")

            with gr.Row():
                survival_score = gr.Number(label="Survival Score", value=0)
                combat_score = gr.Number(label="Combat Score", value=0)
                gathering_score = gr.Number(label="Gathering Score", value=0)

            with gr.Row():
                eval_btn = gr.Button("Run Evaluation")
                eval_results = gr.JSON(label="Evaluation Results")

        with gr.Tab("Curriculum"):
            gr.Markdown("## Curriculum Learning Progress")

            curriculum_info = gr.JSON(label="Current Curriculum Stage")

            with gr.Row():
                advance_btn = gr.Button("Advance Stage")
                regress_btn = gr.Button("Regress Stage")
                stage_result = gr.Textbox(label="Result")

        with gr.Tab("Bot Control"):
            gr.Markdown("## Live Bot Control")

            with gr.Row():
                goal_dropdown = gr.Dropdown(
                    choices=[
                        "basic_movement",
                        "punch_wood",
                        "craft_pickaxe",
                        "mine_stone",
                        "fight_passive",
                        "fight_hostile",
                        "build_shelter",
                        "survive_first_night",
                        "mine_iron",
                        "full_survival",
                        "pvp_combat",
                    ],
                    value="survive_first_night",
                    label="Goal",
                )

            with gr.Row():
                start_btn = gr.Button("Start Bot")
                stop_btn = gr.Button("Stop Bot")
                control_status = gr.Textbox(label="Control Status")

        # ---- Event handlers ----

        def refresh_dashboard():
            progress_data, summary, _metrics, status = get_metrics()
            return summary, status

        refresh_btn.click(fn=refresh_dashboard, outputs=[summary_table, status_text])

        # Auto-refresh training tab
        dashboard.load(
            fn=refresh_dashboard,
            outputs=[summary_table, status_text],
            every=refresh_interval,
        )

        def refresh_benchmarks():
            table_data, summary, history_data, bar_data, status = get_benchmark_data()
            return table_data, summary, history_data, bar_data, status

        bench_refresh_btn.click(
            fn=refresh_benchmarks,
            outputs=[
                bench_table,
                bench_summary,
                bench_overall_plot,
                bench_bar_plot,
                bench_status,
            ],
        )

        # Auto-refresh benchmarks tab
        dashboard.load(
            fn=refresh_benchmarks,
            outputs=[
                bench_table,
                bench_summary,
                bench_overall_plot,
                bench_bar_plot,
                bench_status,
            ],
            every=refresh_interval,
        )

    return dashboard


def main():
    parser = argparse.ArgumentParser(description="Launch training dashboard")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="./checkpoints/logs",
        help="Directory containing metrics.json",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run dashboard on"
    )
    parser.add_argument(
        "--refresh", type=int, default=5, help="Auto-refresh interval in seconds"
    )
    args = parser.parse_args()

    dashboard = create_dashboard(args.metrics_dir, args.refresh)
    dashboard.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
