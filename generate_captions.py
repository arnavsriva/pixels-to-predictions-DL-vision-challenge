"""
One-time script: generates captions for every image using LLaVA-1.5-7B.
LLaVA understands scientific diagrams — much better than BLIP-2 for this task.

Run ONCE before train_with_captions.py:
    modal run generate_captions.py

Output: /results/captions.json  (keys = image_path strings from the CSVs)
~2-3 hours on A10G, ~$3
"""

import modal

app = modal.App("pixels-predictions-captions")

caption_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "accelerate",
        "pillow",
        "tqdm",
        "pandas",
        "numpy",
    )
)

data_volume    = modal.Volume.from_name("scienceqa-dataset", create_if_missing=True)
results_volume = modal.Volume.from_name("scienceqa-results", create_if_missing=True)


@app.function(
    image=caption_image,
    gpu="A10G",           # 24GB VRAM — fits LLaVA-1.5-7B in float16
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=10800,        # 3 hours
    memory=32768,
)
def generate():
    import json
    import warnings
    from pathlib import Path

    import pandas as pd
    from PIL import Image
    from tqdm.auto import tqdm
    import torch

    warnings.filterwarnings("ignore")

    DATA_DIR      = Path("/data")
    IMAGE_DIR     = DATA_DIR / "images" / "images"
    OUTPUT_DIR    = Path("/results")
    CAPTIONS_PATH = OUTPUT_DIR / "captions_llava.json"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Collect every unique image_path from all splits ──────────
    def parse_choices(x):
        import json as _json
        if isinstance(x, list): return x
        if pd.isna(x): return []
        try: return _json.loads(x)
        except Exception:
            import ast; return ast.literal_eval(x)

    all_paths = set()
    for split in ["train.csv", "val.csv", "test.csv"]:
        df = pd.read_csv(DATA_DIR / split)
        for _, row in df.iterrows():
            p = str(row.get("image_path", "")).strip()
            if p and p != "nan":
                all_paths.add(p)
    all_paths = sorted(all_paths)
    print(f"Unique image paths: {len(all_paths):,}")

    # ── Build image index ────────────────────────────────────────
    IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".webp"}
    IMAGE_INDEX = {}
    for p in IMAGE_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            IMAGE_INDEX[p.name] = p
            IMAGE_INDEX[str(p.relative_to(IMAGE_DIR))] = p
            try:
                IMAGE_INDEX[str(p.relative_to(DATA_DIR))] = p
            except ValueError:
                pass

    def load_pil(img_path):
        raw = Path(img_path)
        for c in [DATA_DIR / img_path, IMAGE_DIR / img_path, IMAGE_DIR / raw.name]:
            if c.exists():
                try:
                    img = Image.open(c).convert("RGB")
                    img.thumbnail((512, 512), Image.Resampling.BICUBIC)
                    return img
                except Exception:
                    return None
        for k in [img_path, raw.name]:
            if k in IMAGE_INDEX:
                try:
                    img = Image.open(IMAGE_INDEX[k]).convert("RGB")
                    img.thumbnail((512, 512), Image.Resampling.BICUBIC)
                    return img
                except Exception:
                    return None
        return None

    # ── Load LLaVA-1.5-7B ───────────────────────────────────────
    print("\nLoading LLaVA-1.5-7B...")
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("LLaVA loaded.")

    # Prompt that elicits detailed scientific descriptions
    PROMPT = (
        "USER: <image>\n"
        "Describe this science image in detail. "
        "Include all visible objects, diagrams, labels, arrows, charts, text, "
        "and any scientific concepts illustrated.\n"
        "ASSISTANT:"
    )

    @torch.inference_mode()
    def caption_one(image):
        enc = processor(
            text=PROMPT,
            images=image,
            return_tensors="pt",
        ).to("cuda")
        input_len = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=120,
            do_sample=False,
            temperature=1.0,
        )
        # Decode only new tokens (skip prompt)
        new_tokens = out[0][input_len:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Resume from checkpoint ───────────────────────────────────
    if CAPTIONS_PATH.exists():
        with open(CAPTIONS_PATH) as f:
            captions = json.load(f)
        print(f"Resuming: {len(captions)}/{len(all_paths)} done")
    else:
        captions = {}

    todo    = [p for p in all_paths if p not in captions]
    skipped = 0
    print(f"Images to caption: {len(todo):,}")

    for i, img_path in enumerate(tqdm(todo, desc="Captioning")):
        img = load_pil(img_path)
        if img is None:
            captions[img_path] = ""
            skipped += 1
            continue
        try:
            captions[img_path] = caption_one(img)
        except Exception as e:
            print(f"  Error on {img_path}: {e}")
            captions[img_path] = ""
            skipped += 1

        if (i + 1) % 100 == 0:
            with open(CAPTIONS_PATH, "w") as f:
                json.dump(captions, f)
            results_volume.commit()
            print(f"  Checkpoint: {len(captions)}/{len(all_paths)} | skipped: {skipped}")

    with open(CAPTIONS_PATH, "w") as f:
        json.dump(captions, f)
    results_volume.commit()

    # Show examples
    print(f"\nDone! {len(captions)} captions, {skipped} skipped")
    for k in list(captions.keys())[:3]:
        if captions[k]:
            print(f"\n  Path: {k}")
            print(f"  Caption: {captions[k][:150]}")

    return {"total": len(captions), "skipped": skipped}


@app.local_entrypoint()
def main():
    print("Generating LLaVA-1.5-7B captions for all ScienceQA images...")
    print("~2-3h on A10G GPU")
    result = generate.remote()
    print(f"\nDone: {result['total']} captions, {result['skipped']} skipped")
    print("Now run: modal run train_with_captions.py")
