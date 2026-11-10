
import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.sample.formatters import load_mimic_label_table, normalize_icd9


def load_labels(label_path: str) -> list:
    """Load labels.csv – line index == event token index."""
    try:
        df = pd.read_csv(label_path, sep="\t", header=None)
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).tolist()
    except Exception:
        pass
    df = pd.read_csv(label_path)
    if {"index", "name"}.issubset(df.columns):
        max_idx = int(df["index"].max())
        labels = [""] * (max_idx + 1)
        for _, row in df.iterrows():
            labels[int(row["index"])] = str(row["name"])
        return labels
    return df.iloc[:, 0].astype(str).tolist()


def extract_icd_code(text):
    pattern = r'[A-Z]\d{2}(?=\s|$|[)\],.])' 
    match = re.search(pattern, text)
    return match.group(0) if match else None


def load_mimic_codes(label_path: str, vocab_size: int) -> list[list[str]]:
    """ICD-9 candidates indexed by MIMIC event token; synthetic tokens stay empty."""
    table = load_mimic_label_table(label_path)
    if table and max(table) >= vocab_size:
        raise ValueError("MIMIC vocab_size is smaller than the largest label token ID")
    codes = [[] for _ in range(vocab_size)]
    for token_id, label in table.items():
        for field in ("code_norm", "representative_icd9_code"):
            code = normalize_icd9(label.get(field) or "")
            if re.fullmatch(r"(?:\d{3,5}|[EV]\d{2,4})", code) and code not in codes[token_id]:
                codes[token_id].append(code)
    return codes


def build_embeddings(candidates: list[list[str]], tokenizer) -> tuple[np.ndarray, int]:
    """Embed each token's first supported code, leaving unavailable rows at zero."""
    vectors = {}
    emb_dim = None
    for token_id, codes in enumerate(candidates):
        for code in codes:
            try:
                emb = tokenizer.embed(code)
            except Exception:
                continue
            if emb is None:
                continue
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(emb)
            if not np.isfinite(norm) or norm == 0:
                continue
            if emb_dim is None:
                emb_dim = emb.size
            if emb.size != emb_dim:
                raise ValueError(f"MedTok dimension changed for token {token_id}: {emb.size} != {emb_dim}")
            vectors[token_id] = emb / norm
            break
    if emb_dim is None:
        raise ValueError("No label code could be embedded by MedTok; check the model and code vocabulary")
    embeddings = np.zeros((len(candidates), emb_dim), dtype=np.float32)
    for token_id, vector in vectors.items():
        embeddings[token_id] = vector
    return embeddings, len(vectors)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute MedTok embeddings for UKB or MIMIC event tokens.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data dir.")
    parser.add_argument("--medtok_path", type=str, required=True, help="Path to the MedTok tokenizer model directory.")
    parser.add_argument("--dataset", choices=("ukb", "mimic"), default="ukb")
    parser.add_argument("--vocab_size", type=int, help="Required for MIMIC; includes its visit token ID.")
    args = parser.parse_args()

    if args.dataset == "mimic":
        if not args.vocab_size or args.vocab_size <= 0:
            parser.error("--vocab_size is required for MIMIC")
        label_path = os.path.join(args.data_path, "labels.json")
        candidates = load_mimic_codes(label_path, args.vocab_size)
    else:
        label_path = os.path.join(args.data_path, "labels.csv")
        candidates = [[code] if (code := extract_icd_code(label)) else []
                      for label in load_labels(label_path)]
    print(f"Loaded {len(candidates)} event tokens from {label_path}")

    # 2) Load MedTok tokenizer
    medtok_tokenizer = AutoTokenizer.from_pretrained(
        args.medtok_path, trust_remote_code=True
    )

    embeddings, mapped = build_embeddings(candidates, medtok_tokenizer)
    print(f"Mapped: {mapped}, Skipped (non-ICD / failed): {len(candidates) - mapped}")

    # 5) Save
    output_npy_path = os.path.join(args.data_path, "medtok_embeddings.npy")
    np.save(output_npy_path, embeddings)
    print(f"Saved embeddings to {output_npy_path}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
