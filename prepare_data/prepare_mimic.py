import argparse
import json
import pickle
import random
from pathlib import Path

import pandas as pd


VISIT_TOKEN = "__VISIT_TERMINATE__"


def read_table(input_dir, name, columns):
    path = input_dir / name
    if not path.exists() and path.suffix == ".gz":
        path = path.with_suffix("")
    frame = pd.read_csv(path, compression="infer", dtype={"ICD9_CODE": "string"})
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame[columns]


def sequence(frame, vocabulary, last_visit_days):
    result = []
    for _, patient in frame.groupby("SUBJECT_ID", sort=False):
        patient = patient.sort_values(["ADMITTIME", "HADM_ID"])
        visits = list(patient.itertuples(index=False))
        start = visits[0].ADMITTIME
        items = []
        for index, visit in enumerate(visits):
            base = (visit.ADMITTIME - start).total_seconds() / 86400.0
            interval = (visits[index + 1].ADMITTIME - visit.ADMITTIME).total_seconds() / 86400.0 if index + 1 < len(visits) else last_visit_days
            interval = max(interval, 0.0)
            items.append((base, index, 0, vocabulary[visit.CODE]))
            items.append((base + interval / 2, index, 1, vocabulary[VISIT_TOKEN]))
        items.sort()
        previous = 0.0
        patient_sequence = []
        for time, _, _, event_type in items:
            patient_sequence.append({"time_since_start": float(time), "time_since_last_event": float(time - previous), "type_event": int(event_type)})
            previous = time
        result.append(patient_sequence)
    return result


def prepare(args):
    if not 0 < args.train_fraction < 1 or not 0 <= args.dev_fraction < 1 - args.train_fraction:
        raise ValueError("Invalid train/dev fractions")
    if args.top_k < 0 or args.min_frequency < 1 or not 0 <= args.last_visit_days < float("inf"):
        raise ValueError("Require top_k >= 0, min_frequency >= 1, and finite last_visit_days >= 0")
    admissions = read_table(args.input_dir, "ADMISSIONS.csv.gz", ["SUBJECT_ID", "HADM_ID", "ADMITTIME"])
    diagnoses = read_table(args.input_dir, "DIAGNOSES_ICD.csv.gz", ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    descriptions = read_table(args.input_dir, "D_ICD_DIAGNOSES.csv.gz", ["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"])
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions = admissions.dropna().drop_duplicates(["SUBJECT_ID", "HADM_ID"])
    diagnoses["SEQ_NUM"] = pd.to_numeric(diagnoses["SEQ_NUM"], errors="coerce")
    diagnoses["CODE"] = diagnoses["ICD9_CODE"].str.strip().str.upper().str.replace(".", "", regex=False)
    diagnoses = diagnoses.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "CODE"])
    diagnoses = diagnoses[diagnoses["CODE"].ne("")]
    diagnoses = diagnoses.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "CODE"])
    diagnoses = diagnoses.drop_duplicates(["SUBJECT_ID", "HADM_ID"])
    events = diagnoses.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner", validate="one_to_one")
    if events.empty:
        raise ValueError("No diagnosis visits found")
    subjects = events["SUBJECT_ID"].drop_duplicates().tolist()
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    n_train = round(len(subjects) * args.train_fraction)
    n_dev = round(len(subjects) * args.dev_fraction)
    split_subjects = {"train": subjects[:n_train], "dev": subjects[n_train:n_train + n_dev], "test": subjects[n_train + n_dev:]}
    train = events[events["SUBJECT_ID"].isin(split_subjects["train"])]
    counts = train["CODE"].value_counts()
    counts = counts[counts >= args.min_frequency].head(args.top_k if args.top_k > 0 else None)
    if counts.empty:
        raise ValueError("No diagnosis codes remain after vocabulary filtering")
    vocabulary = {code: index for index, code in enumerate(counts.index)}
    vocabulary[VISIT_TOKEN] = len(vocabulary)
    events = events[events["CODE"].isin(vocabulary)]
    descriptions["CODE"] = descriptions["ICD9_CODE"].str.strip().str.upper().str.replace(".", "", regex=False)
    descriptions = descriptions.drop_duplicates("CODE").set_index("CODE")
    labels = {}
    for code, index in vocabulary.items():
        row = descriptions.loc[code] if code in descriptions.index else None
        labels[str(index)] = {"type_event": index, "code_norm": code, "representative_icd9_code": code, "short_title": "Visit terminate" if code == VISIT_TOKEN else (str(row["SHORT_TITLE"]) if row is not None and pd.notna(row["SHORT_TITLE"]) else code), "long_title": "Visit terminate" if code == VISIT_TOKEN else (str(row["LONG_TITLE"]) if row is not None and pd.notna(row["LONG_TITLE"]) else code)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
    for split, members in split_subjects.items():
        rows = events[events["SUBJECT_ID"].isin(members)]
        sequences = sequence(rows, vocabulary, args.last_visit_days) if not rows.empty else []
        with (args.output_dir / f"{split}.pkl").open("wb") as handle:
            pickle.dump({"dim_process": len(vocabulary), split: sequences}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k", type=int, required=True, help="Use 0 to retain all codes passing min_frequency")
    parser.add_argument("--min_frequency", type=int, required=True)
    parser.add_argument("--train_fraction", type=float, required=True)
    parser.add_argument("--dev_fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--last_visit_days", type=float, required=True)
    prepare(parser.parse_args())
