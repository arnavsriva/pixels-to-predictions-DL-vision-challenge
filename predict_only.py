"""
Inference-only script — loads saved adapter + val tune checkpoint, resumes test prediction.
Run after training: modal run predict_only.py
"""

import modal

app = modal.App("pixels-predictions-infer")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "peft==0.18.1",
        "bitsandbytes==0.45.5",
        "accelerate",
        "pillow",
        "tqdm",
        "pandas",
        "numpy",
    )
)

data_volume = modal.Volume.from_name("scienceqa-dataset")
results_volume = modal.Volume.from_name("scienceqa-results")


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=86400,
    memory=32768,
)
def predict():
    import json
    import random
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm

    import torch
    import torch.nn.functional as F

    warnings.filterwarnings("ignore")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
    DATA_DIR = Path("/data")
    IMAGE_DIR = DATA_DIR / "images" / "images"
    ADAPTER_DIR = Path("/results/smolvlm_scienceqa_lora")
    OUTPUT_DIR = Path("/results")
    IMG_MAX_SIZE = 512
    VAL_TUNE_SAMPLES = 300

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Adapter exists: {ADAPTER_DIR.exists()}")

    # ── Load data ───────────────────────────────────────────────
    def parse_choices(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            return json.loads(x)
        except Exception:
            import ast
            return ast.literal_eval(x)

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    for df in [test_df, val_df]:
        df["choices"] = df["choices"].apply(parse_choices)
    print(f"Test: {len(test_df)} | Val: {len(val_df)}")

    # ── Image loading ───────────────────────────────────────────
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    IMAGE_INDEX = {}
    for p in IMAGE_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            IMAGE_INDEX[p.name] = p
            IMAGE_INDEX[str(p.relative_to(IMAGE_DIR))] = p

    def resolve_image_path(row):
        raw = str(row["image_path"])
        raw_obj = Path(raw)
        for c in [DATA_DIR / raw, IMAGE_DIR / raw, IMAGE_DIR / raw_obj.name]:
            if c.exists():
                return c
        for k in [raw, raw_obj.name]:
            if k in IMAGE_INDEX:
                return IMAGE_INDEX[k]
        raise FileNotFoundError(f"Cannot find: {raw}")

    def load_image(row):
        img = Image.open(resolve_image_path(row)).convert("RGB")
        img.thumbnail((IMG_MAX_SIZE, IMG_MAX_SIZE), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (IMG_MAX_SIZE, IMG_MAX_SIZE), "white")
        canvas.paste(img, ((IMG_MAX_SIZE - img.width) // 2, (IMG_MAX_SIZE - img.height) // 2))
        return canvas

    # ── Prompt — must match train_modal.py exactly ──────────────
    CHOICE_LETTERS = "ABCDEFGHIJ"

    def clean_text(x):
        if x is None or pd.isna(x):
            return ""
        return " ".join(str(x).replace("\n", " ").split())

    def build_prompt(row):
        parts = ["<image>", "Look at the image and answer the science multiple-choice question."]
        subject = clean_text(row.get("subject", ""))
        grade = clean_text(row.get("grade", ""))
        if subject or grade:
            parts.append(f"Subject: {subject}  Grade: {grade}")
        lecture = clean_text(row.get("lecture", ""))
        hint = clean_text(row.get("hint", ""))
        if lecture:
            parts.append(f"Background: {lecture}")
        if hint:
            parts.append(f"Hint: {hint}")
        parts.append(f"Question: {clean_text(row['question'])}")
        choices_text = "\n".join(
            [f"{CHOICE_LETTERS[i]}. {clean_text(c)}" for i, c in enumerate(row["choices"])]
        )
        parts.append(f"Choices:\n{choices_text}")
        parts.append("The correct answer is:")
        return "\n\n".join(parts)

    # ── Load model ──────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    print("Model loaded.")

    # ── Scoring ─────────────────────────────────────────────────
    def move_to_device(batch):
        dev = next(model.parameters()).device
        return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}

    def suffix_letter(row, idx):
        return " " + CHOICE_LETTERS[idx]

    def suffix_sentence(row, idx):
        return " The answer is " + CHOICE_LETTERS[idx] + "."

    def suffix_choice_text(row, idx):
        return " " + CHOICE_LETTERS[idx] + ". " + clean_text(row["choices"][idx])

    @torch.inference_mode()
    def score_nll(row, suffix_fn):
        image = load_image(row)
        prompt = build_prompt(row)
        n = len(row["choices"])
        texts = [prompt + suffix_fn(row, i) for i in range(n)]
        imgs = [image] * n
        enc = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        penc = processor(text=[prompt] * n, images=imgs, padding=True, return_tensors="pt")
        enc = move_to_device(enc)
        out = model(**enc)
        logits = out.logits.float().cpu()
        ids = enc["input_ids"].cpu()
        mask = enc["attention_mask"].cpu()
        pmask = penc["attention_mask"].cpu()
        scores = []
        for i in range(n):
            tlen = int(mask[i].sum().item())
            plen = int(pmask[i].sum().item())
            if plen >= tlen:
                scores.append(1e9)
                continue
            tgt = ids[i, plen:tlen]
            lp = F.log_softmax(logits[i, plen - 1:tlen - 1, :], dim=-1)
            scores.append(-lp[torch.arange(len(tgt)), tgt].mean().item())
        return np.array(scores, dtype=np.float32)

    def normalize(s):
        return (s - s.mean()) / (s.std() + 1e-6)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Get ensemble weights — load from checkpoint or compute
    # ══════════════════════════════════════════════════════════════
    val_ckpt_path = OUTPUT_DIR / "val_tune_checkpoint.json"

    if val_ckpt_path.exists():
        print("\nLoading existing val scores from checkpoint...")
        with open(val_ckpt_path) as f:
            val_cache = json.load(f)
        val_s1 = [np.array(v) for v in val_cache["s1"]]
        val_s2 = [np.array(v) for v in val_cache["s2"]]
        val_s3 = [np.array(v) for v in val_cache["s3"]]
        val_true = val_cache["true"]
        print(f"Loaded {len(val_true)} val scores")
    else:
        print(f"\nScoring {VAL_TUNE_SAMPLES} val samples for weight tuning...")
        val_tune_df = val_df.sample(VAL_TUNE_SAMPLES, random_state=SEED).reset_index(drop=True)
        val_s1, val_s2, val_s3, val_true = [], [], [], []
        for idx in tqdm(range(len(val_tune_df)), desc="Scoring val"):
            row = val_tune_df.iloc[idx]
            val_s1.append(score_nll(row, suffix_letter))
            val_s2.append(score_nll(row, suffix_sentence))
            val_s3.append(score_nll(row, suffix_choice_text))
            val_true.append(int(row["answer"]))
            if (idx + 1) % 50 == 0:
                with open(val_ckpt_path, "w") as f:
                    json.dump({"s1": [s.tolist() for s in val_s1],
                               "s2": [s.tolist() for s in val_s2],
                               "s3": [s.tolist() for s in val_s3],
                               "true": val_true}, f)
                results_volume.commit()
                print(f"  Val checkpoint: {idx+1}/{VAL_TUNE_SAMPLES}")
        with open(val_ckpt_path, "w") as f:
            json.dump({"s1": [s.tolist() for s in val_s1],
                       "s2": [s.tolist() for s in val_s2],
                       "s3": [s.tolist() for s in val_s3],
                       "true": val_true}, f)
        results_volume.commit()

    # Grid search
    val_true_arr = np.array(val_true)
    best_acc, best_w = -1, (1.0, 0.0, 0.0)
    for w1 in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for w2 in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
            for w3 in [0.0, 0.15, 0.25, 0.35, 0.5]:
                preds = [int(np.argmin(w1*normalize(val_s1[i])
                                     + w2*normalize(val_s2[i])
                                     + w3*normalize(val_s3[i])))
                         for i in range(len(val_true))]
                acc = (np.array(preds) == val_true_arr).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_w = (w1, w2, w3)

    W1, W2, W3 = best_w
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Best weights — letter:{W1}, sentence:{W2}, choice_text:{W3}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Predict test set (resumable)
    # ══════════════════════════════════════════════════════════════
    # Uses same checkpoint path as train_modal.py so we resume seamlessly
    test_ckpt_path = OUTPUT_DIR / "test_predictions_checkpoint.json"
    completed = {}
    if test_ckpt_path.exists():
        with open(test_ckpt_path) as f:
            completed = json.load(f)
        print(f"\nResuming test prediction: {len(completed)}/{len(test_df)} done")
    else:
        print("\nStarting fresh test prediction")

    for idx in tqdm(range(len(test_df)), desc="Predicting test"):
        row = test_df.iloc[idx]
        row_id = str(row["id"])
        if row_id in completed:
            continue
        s1 = score_nll(row, suffix_letter)
        s2 = score_nll(row, suffix_sentence)
        s3 = score_nll(row, suffix_choice_text)
        combined = W1*normalize(s1) + W2*normalize(s2) + W3*normalize(s3)
        completed[row_id] = int(np.argmin(combined))
        if len(completed) % 50 == 0:
            with open(test_ckpt_path, "w") as f:
                json.dump(completed, f)
            results_volume.commit()
            print(f"  Test checkpoint: {len(completed)}/{len(test_df)}")

    with open(test_ckpt_path, "w") as f:
        json.dump(completed, f)
    results_volume.commit()

    submission = pd.DataFrame({
        "id": test_df["id"].astype(str),
        "answer": [completed[str(r)] for r in test_df["id"]],
    })

    assert list(submission.columns) == ["id", "answer"]
    assert len(submission) == len(test_df)
    assert submission["id"].is_unique
    for pred, choices in zip(submission["answer"], test_df["choices"]):
        assert 0 <= int(pred) < len(choices)

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    results_volume.commit()
    print(f"\nSaved: {submission_path}")
    print(submission.head(10))
    print(f"\nAnswer distribution:\n{submission['answer'].value_counts().sort_index()}")

    return submission.to_dict()


@app.local_entrypoint()
def main():
    import csv
    print("Running inference only...")
    result = predict.remote()
    local_path = "/Users/arnavsriva/Downloads/pixels-to-predictions/submission.csv"
    with open(local_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for id_, ans in zip(result["id"].values(), result["answer"].values()):
            writer.writerow([id_, ans])
    print(f"Submission saved to: {local_path}")
