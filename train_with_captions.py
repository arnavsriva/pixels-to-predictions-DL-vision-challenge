"""
Train on ALL data (train+val) with LLaVA-1.5-7B image captions in the prompt.

Run AFTER generate_captions.py:
    modal run train_with_captions.py

To train a second model for ensembling, change SEED to 123 and rerun.

What's different from train_all_data.py:
- Loads captions.json (LLaVA-generated) from volume
- Appends "Image description: ..." to every prompt
- 4 epochs instead of 3
- Saves raw NLL scores (not just argmin) so ensemble.py can combine models
"""

import modal

app = modal.App("pixels-predictions-captions-train")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "peft==0.18.1",
        "bitsandbytes==0.45.5",
        "accelerate",
        "datasets",
        "pillow",
        "tqdm",
        "pandas",
        "numpy",
    )
)

data_volume = modal.Volume.from_name("scienceqa-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("scienceqa-results", create_if_missing=True)

TIMEOUT_SECONDS = 8 * 3600


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=TIMEOUT_SECONDS,
    memory=32768,
)
def train_and_predict():
    import gc
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
    from torch.utils.data import Dataset

    warnings.filterwarnings("ignore")

    # ── Change SEED to 123 for second model (ensembling) ────────
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
    DATA_DIR = Path("/data")
    IMAGE_DIR = DATA_DIR / "images" / "images"
    OUTPUT_DIR = Path("/results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Seed-specific dirs so two runs don't overwrite each other
    ADAPTER_DIR = OUTPUT_DIR / f"smolvlm_scienceqa_lora_captions_s{SEED}"

    # ── Hyperparameters (proven config) ─────────────────────────
    IMG_MAX_SIZE = 512
    NUM_EPOCHS = 4       # was 3 — one more epoch for extra signal
    PER_DEVICE_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8
    LEARNING_RATE = 1.5e-4
    WARMUP_RATIO = 0.05
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 10
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load captions ────────────────────────────────────────────
    captions_path = OUTPUT_DIR / "captions_llava.json"
    if not captions_path.exists():
        raise FileNotFoundError(
            "captions_llava.json not found! Run generate_captions.py first:\n"
            "  modal run generate_captions.py"
        )
    with open(captions_path) as f:
        CAPTIONS = json.load(f)
    valid_captions = sum(1 for v in CAPTIONS.values() if v.strip())
    print(f"Loaded {len(CAPTIONS):,} captions ({valid_captions:,} non-empty)")

    # ── Load data ────────────────────────────────────────────────
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

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    for df in [train_df, val_df, test_df]:
        df["choices"] = df["choices"].apply(parse_choices)

    all_train_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Combined: {len(all_train_df):,} | Test: {len(test_df):,}")

    # ── Image handling ───────────────────────────────────────────
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    IMAGE_INDEX = {}
    for p in IMAGE_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            IMAGE_INDEX[p.name] = p
            IMAGE_INDEX[str(p.relative_to(IMAGE_DIR))] = p
            try:
                IMAGE_INDEX[str(p.relative_to(DATA_DIR))] = p
            except ValueError:
                pass
    print(f"Indexed images: {len(IMAGE_INDEX)}")

    def resolve_image_path(row):
        raw_path = str(row["image_path"])
        raw_path_obj = Path(raw_path)
        for c in [DATA_DIR / raw_path, IMAGE_DIR / raw_path, IMAGE_DIR / raw_path_obj.name]:
            if c.exists():
                return c
        for key in [raw_path, raw_path_obj.name]:
            if key in IMAGE_INDEX:
                return IMAGE_INDEX[key]
        raise FileNotFoundError(f"Could not find image: {raw_path}")

    def load_image(row):
        img = Image.open(resolve_image_path(row)).convert("RGB")
        img.thumbnail((IMG_MAX_SIZE, IMG_MAX_SIZE), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (IMG_MAX_SIZE, IMG_MAX_SIZE), "white")
        canvas.paste(img, ((IMG_MAX_SIZE - img.width) // 2, (IMG_MAX_SIZE - img.height) // 2))
        return canvas

    # ── Prompt (with caption) ────────────────────────────────────
    CHOICE_LETTERS = "ABCDEFGHIJ"

    def clean_text(x):
        if x is None or pd.isna(x):
            return ""
        return " ".join(str(x).replace("\n", " ").split())

    def get_caption(row):
        img_path = str(row.get("image_path", "")).strip()
        if not img_path or img_path == "nan":
            return ""
        cap = CAPTIONS.get(img_path, "")
        return clean_text(cap)

    def build_prompt(row, include_answer=False):
        parts = ["<image>", "Look at the image and answer the science multiple-choice question."]

        subject = clean_text(row.get("subject", ""))
        grade = clean_text(row.get("grade", ""))
        if subject or grade:
            parts.append(f"Subject: {subject}  Grade: {grade}")

        # Caption: explicit visual description from BLIP-2
        caption = get_caption(row)
        if caption:
            parts.append(f"Image description: {caption}")

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

        prompt = "\n\n".join(parts)
        if include_answer:
            prompt += " " + CHOICE_LETTERS[int(row["answer"])]
        return prompt

    # ── Dataset ──────────────────────────────────────────────────
    class ScienceQADataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return {"row": row, "image": load_image(row)}

    # ── Model ────────────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params <= 5_000_000, f"Exceeded 5M cap: {trainable_params:,}"

    # ── Collator ─────────────────────────────────────────────────
    def train_collate_fn(batch):
        images = [b["image"] for b in batch]
        prompts = [build_prompt(b["row"], include_answer=False) for b in batch]
        answer_texts = [" " + CHOICE_LETTERS[int(b["row"]["answer"])] for b in batch]
        eos = processor.tokenizer.eos_token or ""
        full_texts = [p + a + eos for p, a in zip(prompts, answer_texts)]

        full = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_only = processor(text=prompts, images=images, padding=True, return_tensors="pt")

        labels = full["input_ids"].clone()
        labels[full["attention_mask"] == 0] = -100
        for i in range(len(batch)):
            prompt_len = int(prompt_only["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        full["labels"] = labels
        return full

    # ══════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"TRAINING: {len(all_train_df):,} samples + captions, {NUM_EPOCHS} epochs, seed={SEED}")
    print("=" * 60)

    from transformers import Trainer, TrainingArguments, TrainerCallback

    def move_to_device(mdl, batch):
        dev = next(mdl.parameters()).device
        return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}

    CHECKPOINT_DIR = OUTPUT_DIR / f"checkpoints_captions_s{SEED}"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            (ckpt_dir / "plain_lora.txt").write_text("true")
            results_volume.commit()
            print(f"  [Volume committed at step {state.global_step}]")

    existing_checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
    resume_from = None
    if existing_checkpoints:
        ckpt = existing_checkpoints[-1]
        if (ckpt / "plain_lora.txt").exists():
            resume_from = str(ckpt)
            print(f"Resuming from: {resume_from}")
        else:
            import shutil
            for c in existing_checkpoints:
                shutil.rmtree(c, ignore_errors=True)
            print("Cleared incompatible checkpoints")
    if not resume_from:
        print("Starting training from scratch")

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ScienceQADataset(all_train_df),
        data_collator=train_collate_fn,
        callbacks=[VolumeCommitCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    results_volume.commit()
    print(f"Saved adapter to: {ADAPTER_DIR}")

    # ══════════════════════════════════════════════════════════════
    # PREDICT TEST — saves raw NLL scores for ensembling
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PREDICTING: test set, seed={SEED} (saving raw NLL scores)")
    print("=" * 60)

    model.eval()

    @torch.inference_mode()
    def score_nll(row):
        """Returns raw NLL scores for every choice (lower = more likely)."""
        image  = load_image(row)
        prompt = build_prompt(row)   # includes LLaVA caption
        n      = len(row["choices"])
        texts  = [prompt + " " + CHOICE_LETTERS[i] for i in range(n)]
        imgs   = [image] * n

        enc  = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        penc = processor(text=[prompt]*n, images=imgs, padding=True, return_tensors="pt")

        dev = next(model.parameters()).device
        enc = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in enc.items()}

        out    = model(**enc)
        logits = out.logits.float().cpu()
        ids    = enc["input_ids"].cpu()
        mask   = enc["attention_mask"].cpu()
        pmask  = penc["attention_mask"].cpu()

        scores = []
        for i in range(n):
            tlen = int(mask[i].sum().item())
            plen = int(pmask[i].sum().item())
            if plen >= tlen:
                scores.append(1e9)
                continue
            tgt = ids[i, plen:tlen]
            lp  = F.log_softmax(logits[i, plen-1:tlen-1, :], dim=-1)
            scores.append(-lp[torch.arange(len(tgt)), tgt].mean().item())
        return scores   # list of floats, one per choice

    # Seed-specific score file so two model runs don't clash
    scores_path = OUTPUT_DIR / f"test_nll_scores_captions_s{SEED}.json"
    all_scores  = {}   # {row_id: [score_A, score_B, ...]}
    if scores_path.exists():
        with open(scores_path) as f:
            all_scores = json.load(f)
        print(f"Resuming: {len(all_scores)}/{len(test_df)} done")
    else:
        print("Starting fresh")

    for idx in tqdm(range(len(test_df)), desc="Scoring test"):
        row    = test_df.iloc[idx]
        row_id = str(row["id"])
        if row_id in all_scores:
            continue
        all_scores[row_id] = score_nll(row)
        if len(all_scores) % 50 == 0:
            with open(scores_path, "w") as f:
                json.dump(all_scores, f)
            results_volume.commit()
            print(f"  Checkpoint: {len(all_scores)}/{len(test_df)}")

    with open(scores_path, "w") as f:
        json.dump(all_scores, f)
    results_volume.commit()
    print(f"Raw NLL scores saved to: {scores_path}")

    # Also build a standalone submission from this model alone
    completed = {rid: int(np.argmin(s)) for rid, s in all_scores.items()}

    submission = pd.DataFrame({
        "id":     test_df["id"].astype(str),
        "answer": [completed[str(r)] for r in test_df["id"]],
    })

    assert list(submission.columns) == ["id", "answer"]
    assert len(submission) == len(test_df)
    assert submission["id"].is_unique
    for pred, choices in zip(submission["answer"], test_df["choices"]):
        assert 0 <= int(pred) < len(choices)

    submission_path = OUTPUT_DIR / f"submission_captions_s{SEED}.csv"
    submission.to_csv(submission_path, index=False)
    results_volume.commit()
    print(f"\nSaved: {submission_path}")
    print(submission.head(10))
    print(f"\nAnswer distribution:\n{submission['answer'].value_counts().sort_index()}")

    return submission.to_dict()


@app.local_entrypoint()
def main():
    print("Training with LLaVA captions (4 epochs)...")
    print("~4.5h training + ~1.5h prediction = ~6h total")
    result = train_and_predict.remote()

    import csv, json as _json
    seed = SEED   # automatically matches whichever SEED was used above
    local_path = f"/Users/arnavsriva/Downloads/pixels-to-predictions/submission_captions_s{seed}.csv"
    with open(local_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for id_, ans in zip(result["id"].values(), result["answer"].values()):
            writer.writerow([id_, ans])
    print(f"Submission saved to: {local_path}")
    print("To ensemble: run again with SEED=123 then: modal run ensemble.py")
