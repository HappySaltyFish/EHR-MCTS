"""PRM prefix examples, label alignment and MedTok features for training."""

import json
from reward_model.prompts import RAG_SYSTEM_PROMPT, ORM_SYSTEM_PROMPT, PRM_SYSTEM_PROMPT
from reward_model.medtok_features import pair_features, solution_prefixes
from reward_model.rag_prompt import build_prm_prompt


def normalize_train_json_paths(train_json):
    if isinstance(train_json, str): return [train_json]
    elif isinstance(train_json, (list, tuple)): return list(train_json)
    else: raise TypeError(f"train_json must be a string or a list/tuple of strings, got {type(train_json)}")


def process_gemini_label(label_list):
    processed, found_zero = [], False
    for v in label_list:
        processed.append(0 if found_zero else v)
        if v == 0:
            found_zero = True
    return processed


def format_question_with_options(item):
    return item.get("question", "")


def process_all_results_to_dataset(
    all_results, tokenizer, step_tag_id=128256, medtok_emb_table=None,
    max_token_len=None,
):
    data = {"input_ids": [], "attention_mask": [], "labels": [], "values": []}
    data["history_embeddings"] = []
    data["prefix_embeddings"] = []
    errors = 0
    for entry in all_results:
        raw = entry["query"].replace(" ки\n", " ки")  # Clean line breaks
        enc = tokenizer(raw, add_special_tokens=True, truncation=True)

        ids = enc["input_ids"] + [tokenizer.pad_token_id]
        attn = enc["attention_mask"] + [0]

        raw_label = raw.replace(" ки", " +")
        ref = tokenizer(raw_label, add_special_tokens=True)["input_ids"]
        ref.insert(0, tokenizer.pad_token_id)
        ref[0] = -100

        vals, cnt, ann = [], 0, entry["annotation"]
        if len(ref) != len(ids):
            errors += 1
            continue

        skip = False
        for j in range(len(ref)):
            if j == 0:
                vals.append(0)
                continue
            if ids[j - 1] == step_tag_id:
                if cnt < len(ann):
                    vals.append(ann[cnt]); cnt += 1
                else:
                    errors += 1; skip = True; break
            else:
                vals.append(0); ref[j] = -100
        if skip or cnt != len(ann):
            continue

        target_marker = entry.get("target_marker_index")
        if target_marker is not None:
            if target_marker >= cnt:
                errors += 1
                continue
            seen = 0
            for j in range(len(ref)):
                if ref[j] != -100:
                    if seen != target_marker:
                        ref[j] = -100
                    seen += 1
        if max_token_len is not None and not any(
            label != -100 for label in ref[:max_token_len]
        ):
            errors += 1
            continue

        data["input_ids"].append(ids)
        data["attention_mask"].append(attn)
        data["labels"].append(ref)
        data["values"].append(vals)

        features = pair_features(
            entry.get("history_tokens", []), entry.get("prefix_tokens", []),
            medtok_emb_table,
        )
        for key, value in features.items():
            data[key].append(value)

    print(f"Skipped due to errors: {errors} solution(s)")
    return data


def build_dataset_from_json(
    data, tokenizer, step_tag_id: int, args, medtok_emb_table=None,
):
    """Convert sequence prediction samples to LLM input."""
    if args.use_rag == "yes":
        SYSTEM_PROMPT = RAG_SYSTEM_PROMPT
    elif args.train_label == "orm_label":
        SYSTEM_PROMPT = ORM_SYSTEM_PROMPT
    else:
        SYSTEM_PROMPT = PRM_SYSTEM_PROMPT

    all_results = []

    for idx, item in enumerate(data):
        qid = item.get("question_id", f"q_{idx}")
        q_formatted = format_question_with_options(item)
        sols = item.get("solutions", [])
        if not sols:
            continue

        for sidx, sol in enumerate(sols):
            solution_tokens = sol.get("solution_tokens", [])
            if args.use_rag == "yes":
                docs_by_step = sol.get("related_docs_by_step")
                if (sol.get("rag_policy") != "key_overlap_prefix_v1"
                        or not isinstance(docs_by_step, list)
                        or len(docs_by_step) != len(solution_tokens)
                        or any(not isinstance(docs, list)
                               or any(not isinstance(doc, str) for doc in docs)
                               for docs in docs_by_step)):
                    raise ValueError(f"Strict per-prefix RAG documents are required for {qid} solution {sidx}")
            else:
                docs_by_step = []

            # Label extraction
            if args.train_label == "gemini_label":
                arr = sol.get("prm_gemini_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "llama_label":
                arr = sol.get("prm_llama_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "orm_label":
                label = sol.get("orm_label", [])
                label = label if isinstance(label, list) else [label]
            else:
                label = sol.get(args.train_label, [])
                label = label if isinstance(label, list) else [label]

            prefixes = solution_prefixes(sol.get("prm_processed_solution", ""))
            if len(prefixes) != len(solution_tokens) or len(prefixes) != len(label):
                raise ValueError(f"Marker/token/label count mismatch for {qid} solution {sidx}")
            if "input_tokens" not in item:
                raise ValueError(f"Missing input_tokens for {qid}; history MedTok summary cannot be inferred from text")
            for step, prefix_text in enumerate(prefixes):
                raw_text = build_prm_prompt(
                    tokenizer, SYSTEM_PROMPT, q_formatted, prefix_text,
                    args.max_token_len,
                    docs_by_step[step] if args.use_rag == "yes" else (),
                )
                all_results.append({
                    "query": raw_text,
                    "annotation": label[:step + 1],
                    "target_marker_index": step,
                    "question_id": qid,
                    "solution_index": sidx,
                    "history_tokens": item["input_tokens"],
                    "prefix_tokens": solution_tokens[:step + 1],
                })

    return process_all_results_to_dataset(
        all_results, tokenizer, step_tag_id=step_tag_id,
        medtok_emb_table=medtok_emb_table,
        max_token_len=args.max_token_len,
    )


def split_json_data(json_paths, args):
    json_paths = normalize_train_json_paths(json_paths)

    data = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            _data = json.load(f)
        if not isinstance(_data, list): raise ValueError(f"JSON file must contain a list, but got {type(_data)} from {json_path}")
        print(f"Loaded {len(_data)} samples from json_path={json_path}")
        data.extend(_data)
    print(f"Total merged data count: {len(data)} from {len(json_paths)} file(s)")

    if args.do_filtering.lower() == "yes":
        for item in data:
            if "solutions" not in item:
                continue

            solutions = item["solutions"]

            orm_0_solutions = []
            orm_1_solutions = []

            for sol in solutions:
                orm_label_val = sol.get("orm_label", 0)
                if orm_label_val == 0:
                    if args.train_label == "er_label":
                        orm_0_solutions.append(sol)
                        continue

                    if args.train_label == "gemini_label":
                        arr = sol.get("prm_gemini_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    elif args.train_label == "llama_label":
                        arr = sol.get("prm_llama_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    else:
                        arr = sol.get(args.train_label, [])
                        if not isinstance(arr, list):
                            arr = [arr]

                    if any(x == 0 for x in arr):
                        orm_0_solutions.append(sol)
                else:
                    orm_1_solutions.append(sol)

            remain_0_count = len(orm_0_solutions)
            need_1_count = max(remain_0_count, 2)

            keep_1_solutions = orm_1_solutions[:need_1_count]

            item["solutions"] = orm_0_solutions + keep_1_solutions

    return data, []
