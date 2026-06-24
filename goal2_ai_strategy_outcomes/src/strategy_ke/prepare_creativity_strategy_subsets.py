import os
from pathlib import Path

import pandas as pd


BASE = Path(os.environ["GENAI_DATA_ROOT"])
ROOT = Path(os.environ.get("GENAI_ANALYSIS_ROOT", Path.cwd()))
UPSTREAM = ROOT / "upstream_repo" / "multi-turn-analysis-master" / "out" / "final_cvae"
CREATIVITY = (
    BASE
    / "07_multiclass_creativity_classifier"
    / "03_full_dataset_predictions"
    / "full_data_multiclass_conversation_summary.csv"
)
DATA = ROOT / "01_analysis_data"
DATA.mkdir(parents=True, exist_ok=True)


def main():
    trajectories = pd.read_csv(UPSTREAM / "conversation_trajectory_summaries.csv.gz")
    creativity = pd.read_csv(CREATIVITY)

    assert trajectories["conversation_hash"].is_unique
    assert creativity["conversation_hash"].is_unique

    merged = trajectories.merge(
        creativity[
            [
                "conversation_hash",
                "n_turns",
                "broad_creativity_turns",
                "strict_brainstorm_turns",
                "first_multiclass_label",
                "has_broad_creativity_v2",
                "has_strict_brainstorm_v2",
                "broad_creativity_turn_share",
                "strict_brainstorm_turn_share",
            ]
        ],
        on="conversation_hash",
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == len(trajectories) == len(creativity)

    broad = merged.loc[merged["has_broad_creativity_v2"]].copy()
    strict = merged.loc[merged["has_strict_brainstorm_v2"]].copy()

    broad_path = DATA / "broad_creativity_conversation_trajectories.csv"
    strict_path = DATA / "strict_brainstorm_conversation_trajectories.csv"
    merged_summary_path = DATA / "strategy_creativity_merge_summary.csv"

    broad.to_csv(broad_path, index=False)
    strict.to_csv(strict_path, index=False)
    pd.DataFrame(
        [
            {
                "sample": "all_conversations",
                "conversations": len(merged),
                "mean_KE": merged["ke_overall_ke_mean"].mean(),
                "mean_SAT": merged["satisfaction_SAT"].mean(),
            },
            {
                "sample": "broad_creativity",
                "conversations": len(broad),
                "mean_KE": broad["ke_overall_ke_mean"].mean(),
                "mean_SAT": broad["satisfaction_SAT"].mean(),
            },
            {
                "sample": "strict_brainstorming",
                "conversations": len(strict),
                "mean_KE": strict["ke_overall_ke_mean"].mean(),
                "mean_SAT": strict["satisfaction_SAT"].mean(),
            },
        ]
    ).to_csv(merged_summary_path, index=False)

    print(broad_path)
    print(strict_path)
    print(merged_summary_path)
    print(f"broad={len(broad):,}; strict={len(strict):,}; all={len(merged):,}")


if __name__ == "__main__":
    main()
