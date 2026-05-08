"""
Ensemble predictions from multiple trained models by averaging raw NLL scores.

Prerequisites — run these first (change SEED between runs):
    SEED=42:  modal run train_with_captions.py   → test_nll_scores_captions_s42.json
    SEED=123: modal run train_with_captions.py   → test_nll_scores_captions_s123.json

Then:
    modal run ensemble.py

Averaging NLL scores is better than majority voting because it uses
confidence — a model that's 90% sure of A beats two models 51% sure of B.
"""

import modal

app = modal.App("pixels-predictions-ensemble")

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy")
)

data_volume    = modal.Volume.from_name("scienceqa-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("scienceqa-results", create_if_missing=True)


@app.function(
    image=cpu_image,
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=600,
)
def ensemble():
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    DATA_DIR   = Path("/data")
    OUTPUT_DIR = Path("/results")

    # ── Load test set ────────────────────────────────────────────
    def parse_choices(x):
        if isinstance(x, list): return x
        if pd.isna(x): return []
        try: return json.loads(x)
        except Exception:
            import ast; return ast.literal_eval(x)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["choices"] = test_df["choices"].apply(parse_choices)
    print(f"Test samples: {len(test_df):,}")

    # ── Load NLL score files ─────────────────────────────────────
    score_files = sorted(OUTPUT_DIR.glob("test_nll_scores_captions_s*.json"))
    if len(score_files) < 2:
        raise FileNotFoundError(
            f"Found {len(score_files)} score file(s), need at least 2.\n"
            "Run train_with_captions.py with SEED=42 AND SEED=123 first."
        )

    print(f"Ensembling {len(score_files)} models:")
    all_model_scores = []
    for path in score_files:
        with open(path) as f:
            scores = json.load(f)
        print(f"  {path.name}: {len(scores)} samples")
        all_model_scores.append(scores)

    # Verify all models scored all test samples
    for scores in all_model_scores:
        assert len(scores) == len(test_df), f"Score count mismatch: {len(scores)} vs {len(test_df)}"

    # ── Average NLL scores across models ─────────────────────────
    # Lower NLL = model thinks this choice is more likely to be correct
    # Average scores, then pick argmin
    predictions = {}
    for _, row in test_df.iterrows():
        row_id = str(row["id"])
        n      = len(row["choices"])

        # Collect scores from each model, normalize per-model to handle scale differences
        model_arrays = []
        for scores in all_model_scores:
            s = np.array(scores[row_id], dtype=np.float32)
            # Normalize: subtract mean so each model contributes equally
            s = s - s.mean()
            model_arrays.append(s)

        avg = np.mean(model_arrays, axis=0)
        predictions[row_id] = int(np.argmin(avg))

    # ── Build submission ─────────────────────────────────────────
    submission = pd.DataFrame({
        "id":     test_df["id"].astype(str),
        "answer": [predictions[str(r)] for r in test_df["id"]],
    })

    assert list(submission.columns) == ["id", "answer"]
    assert len(submission) == len(test_df)
    assert submission["id"].is_unique
    for pred, choices in zip(submission["answer"], test_df["choices"]):
        assert 0 <= int(pred) < len(choices)

    n_models = len(score_files)
    out_path  = OUTPUT_DIR / f"submission_ensemble_{n_models}models.csv"
    submission.to_csv(out_path, index=False)
    results_volume.commit()

    print(f"\nEnsemble of {n_models} models saved: {out_path}")
    print(f"\nAnswer distribution:\n{submission['answer'].value_counts().sort_index()}")

    return submission.to_dict()


@app.local_entrypoint()
def main():
    print("Ensembling model predictions...")
    result = ensemble.remote()

    import csv
    local_path = "/Users/arnavsriva/Downloads/pixels-to-predictions/submission_ensemble.csv"
    with open(local_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for id_, ans in zip(result["id"].values(), result["answer"].values()):
            writer.writerow([id_, ans])
    print(f"Ensemble submission saved to: {local_path}")
