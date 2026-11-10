import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(input_dir, name, columns):
    path = input_dir / name
    frame = pd.read_csv(path, dtype={"eid": "string"})
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def event_rows(frame, dates, codes):
    date = pd.to_datetime(frame[dates], errors="coerce")
    code = frame[codes].astype("string").str.strip().str.upper().str.replace(".", "", regex=False)
    valid = date.notna() & code.str.fullmatch(r"[A-Z][0-9]{2}[A-Z0-9]*", na=False)
    valid &= ~date.isin(pd.to_datetime(["1900-01-01", "1901-01-01"]))
    return pd.DataFrame({"eid": frame.loc[valid, "eid"], "date": date[valid], "token": code[valid]})


def prepare(args):
    if args.min_events < 1 or not 0 < args.train_fraction < 1:
        raise ValueError("Require min_events >= 1 and 0 < train_fraction < 1")
    baseline = read_table(args.input_dir, "baseline.csv", ["eid", "birth_year", "birth_month", "sex", "bmi", "smoking", "alcohol"])
    first = read_table(args.input_dir, "first_occurrence.csv", ["eid", "date", "icd10"])
    cancer = read_table(args.input_dir, "cancer.csv", ["eid", "date", "icd10"])
    death = read_table(args.input_dir, "death.csv", ["eid", "date"])
    baseline = baseline.drop_duplicates("eid").copy()
    month = pd.to_numeric(baseline["birth_month"], errors="coerce")
    named = pd.to_datetime(baseline["birth_month"], format="%B", errors="coerce").dt.month
    short = pd.to_datetime(baseline["birth_month"], format="%b", errors="coerce").dt.month
    month = month.fillna(named).fillna(short)
    dob = pd.to_datetime(pd.DataFrame({"year": pd.to_numeric(baseline["birth_year"], errors="coerce"), "month": month, "day": 1}), errors="coerce")
    baseline["dob"] = dob
    baseline = baseline.dropna(subset=["eid", "dob"])
    baseline["eid"] = pd.to_numeric(baseline["eid"], errors="coerce")
    baseline = baseline.dropna(subset=["eid"])
    baseline["eid"] = baseline["eid"].astype("uint64")
    bmi = pd.to_numeric(baseline["bmi"], errors="coerce")
    baseline["bmi_bucket"] = np.select([bmi > 28, bmi > 22, bmi.notna()], [5, 4, 3], default=0)
    parts = []
    smoking = {"no": "NO", "only occasionally": "OCCASIONAL", "yes, on most or all days": "MOST_DAYS", "prefer not to answer": "PNA"}
    alcohol = {"never": "NEVER", "special occasions only": "SPECIAL", "one to three times a month": "MONTHLY_1_3", "once or twice a week": "WEEKLY_1_2", "three or four times a week": "WEEKLY_3_4", "daily or almost daily": "DAILY", "prefer not to answer": "PNA"}
    baseline["smoking"] = baseline["smoking"].astype("string").str.strip().str.lower().map(smoking).fillna("MISSING")
    baseline["alcohol"] = baseline["alcohol"].astype("string").str.strip().str.lower().map(alcohol).fillna("MISSING")
    for column in ("sex", "bmi_bucket", "smoking", "alcohol"):
        values = baseline[column].astype("string").str.strip().str.upper()
        values = values.str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
        valid = values.notna() & values.ne("")
        if column == "bmi_bucket":
            valid &= values.ne("0")
        name = {"sex": "p31", "bmi_bucket": "bmi_status", "smoking": "smoking_status", "alcohol": "alcohol_status"}[column]
        parts.append(pd.DataFrame({"eid": baseline.loc[valid, "eid"], "age_days": 0, "token": "BL__" + name + "__" + values[valid]}))
    events = pd.concat([event_rows(first, "date", "icd10"), event_rows(cancer, "date", "icd10")], ignore_index=True)
    events["eid"] = pd.to_numeric(events["eid"], errors="coerce")
    events = events.dropna(subset=["eid"])
    events["eid"] = events["eid"].astype("uint64")
    events = events.merge(baseline[["eid", "dob"]], on="eid", how="inner")
    events["age_days"] = (events["date"] - events["dob"]).dt.days
    events = events.loc[events["age_days"].ge(0), ["eid", "age_days", "token"]]
    all_events = pd.concat(parts + [events], ignore_index=True)
    death_events = death[["eid", "date"]].copy()
    death_events["eid"] = pd.to_numeric(death_events["eid"], errors="coerce")
    death_events["date"] = pd.to_datetime(death_events["date"], errors="coerce")
    death_events = death_events.dropna().sort_values("date").drop_duplicates("eid")
    death_events["eid"] = death_events["eid"].astype("uint64")
    death_events = death_events.merge(baseline[["eid", "dob"]], on="eid", how="inner")
    death_events["age_days"] = (death_events["date"] - death_events["dob"]).dt.days
    death_events = death_events.loc[death_events["age_days"].ge(0), ["eid", "age_days"]]
    death_events["token"] = "Death."
    if not death_events.empty:
        all_events = pd.concat([all_events, death_events], ignore_index=True)
    all_events = all_events[all_events["age_days"].between(0, np.iinfo(np.uint32).max)]
    all_events = all_events[all_events["eid"].between(0, np.iinfo(np.uint32).max)]
    counts = all_events.groupby("eid").size()
    all_events = all_events[all_events["eid"].isin(counts[counts >= args.min_events].index)]
    if all_events.empty:
        raise ValueError("No patients remain after min_events filtering")
    tokens = sorted(set(all_events["token"]) - {"Death."}) + ["Death."]
    token_to_id = {token: index + 1 for index, token in enumerate(tokens)}
    all_events["token_id"] = all_events["token"].map(token_to_id)
    all_events = all_events.sort_values(["eid", "age_days", "token_id"])
    eids = all_events["eid"].drop_duplicates().to_numpy(dtype=np.uint32)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(eids)
    train_eids = eids[:int(len(eids) * args.train_fraction)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"token": tokens}).to_csv(args.output_dir / "labels.csv", index=False)
    baseline[baseline["eid"].isin(eids)].to_csv(args.output_dir / "participant_baseline.csv", index=False)
    for split, members in (("train", train_eids), ("val", eids[len(train_eids):])):
        rows = all_events[all_events["eid"].isin(members)]
        rows[["eid", "age_days", "token_id"]].to_numpy(dtype=np.uint32).tofile(args.output_dir / f"{split}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_events", type=int, required=True)
    parser.add_argument("--train_fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    prepare(parser.parse_args())
