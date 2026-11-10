import os
import pickle
import numpy as np
import torch
import re
import random
import math


# =============================================================================
# LLM API settings
# =============================================================================
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
# DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_SETTINGS = {
    "deepseek": {
        "API_KEY": "",
        "BASE_URL": "https://api.deepseek.com/v1",
        "MODEL": "deepseek-chat",
    },
    "openai": {
        "API_KEY": OPENAI_API_KEY,
        "BASE_URL": OPENAI_BASE_URL,
        "MODEL": OPENAI_MODEL,
    },
}


def get_llm_settings(name: str):
    key = str(name).strip().lower()
    if key not in LLM_SETTINGS:
        raise KeyError(f"Unknown LLM settings group: {name!r}")
    return LLM_SETTINGS[key]


def get_p2i(data):
    """
    Get the patient to index mapping.
    p2i[a] = (b,c) means that person a's data is in data[b:b+c]
    """

    print(f"get_p2i: data shape: {data.shape}")
    print(f"get_p2i: sample data: {data[:10,:]}")

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def infer_nhp_split_key(path, split=None):
    """Infer the payload key used by NHP-style pkl files."""
    if split is not None:
        return "dev" if split == "val" else split

    stem = os.path.splitext(os.path.basename(path))[0]
    return "dev" if stem == "val" else stem


def load_nhp_pkl_split(path, split=None):
    split_key = infer_nhp_split_key(path, split)
    with open(path, "rb") as f:
        payload = pickle.load(f, encoding="latin-1")

    if split_key not in payload:
        raise KeyError(f"{path} does not contain split key {split_key!r}; keys={list(payload.keys())}")
    if "dim_process" not in payload:
        raise KeyError(f"{path} does not contain required key 'dim_process'")

    return payload[split_key], int(payload["dim_process"])


def nhp_sequences_to_delphi_rows(
    sequences,
    patient_id_offset: int = 0,
    time_scale: float = 1.0,
) -> np.ndarray:
    rows = []
    order_keys = []
    if time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")

    for patient_idx, seq in enumerate(sequences):
        patient_id = patient_id_offset + patient_idx
        for event_idx, event in enumerate(seq):
            try:
                time_since_start = float(event["time_since_start"])
                type_event = int(event["type_event"])
            except KeyError as exc:
                raise KeyError(f"NHP event is missing required key {exc.args[0]!r}: {event}") from exc

            if time_since_start < 0:
                raise ValueError(f"time_since_start must be non-negative, got {time_since_start}")
            if type_event < 0:
                raise ValueError(f"type_event must be non-negative, got {type_event}")

            scaled_time = int(round(time_since_start * time_scale))
            rows.append((patient_id, scaled_time, type_event))
            order_keys.append((patient_id, scaled_time, event_idx))

    if not rows:
        return np.empty((0, 3), dtype=np.uint32)

    out = np.asarray(rows, dtype=np.uint64)
    if out.max() > np.iinfo(np.uint32).max:
        raise OverflowError("Converted NHP rows exceed uint32 range; reduce time_scale or check input data.")

    order_key_arr = np.asarray(order_keys, dtype=np.uint64)
    order = np.lexsort((order_key_arr[:, 2], order_key_arr[:, 1], order_key_arr[:, 0]))
    return out[order].astype(np.uint32, copy=False)


def save_delphi_bin(rows, output_path):
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"Expected rows with shape (N, 3), got {rows.shape}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    rows.astype(np.uint32, copy=False).reshape(-1).tofile(output_path)


def convert_nhp_pkl_split_to_delphi_bin(
    input_path,
    output_path,
    split=None,
    patient_id_offset=0,
    time_scale=1.0,
):
    sequences, dim_process = load_nhp_pkl_split(input_path, split=split)
    rows = nhp_sequences_to_delphi_rows(
        sequences,
        patient_id_offset=patient_id_offset,
        time_scale=time_scale,
    )
    save_delphi_bin(rows, output_path)
    return int(rows.shape[0]), dim_process


def convert_nhp_pkl_dir_to_delphi_bin(
    input_dir,
    output_dir,
    time_scale=1.0,
    overwrite=False,
):
    split_specs = [
        ("train", "train.pkl", "train.bin"),
        ("dev", "dev.pkl", "val.bin"),
        ("test", "test.pkl", "test.bin"),
    ]
    summary = {}

    for split_key, input_name, output_name in split_specs:
        input_path = os.path.join(input_dir, input_name)
        if not os.path.exists(input_path):
            continue

        output_path = os.path.join(output_dir, output_name)
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")

        num_events, dim_process = convert_nhp_pkl_split_to_delphi_bin(
            input_path=input_path,
            output_path=output_path,
            split=split_key,
            time_scale=time_scale,
        )
        summary[split_key] = {
            "num_events": num_events,
            "dim_process": dim_process,
        }

    if not summary:
        raise FileNotFoundError(f"No train/dev/test pkl files found in {input_dir}")

    return summary


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False, 
              no_event_token_rate=5, cut_batch=False):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'center', 'right', 'smart_random', 'smart_right'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            #TODO maybe use the same shift for all lifestyle tokens in the trajectory?
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    elif padding == 'nopad':
        pass
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values

    if padding != 'nopad':
        # stack "no event" tokens with real tokens
        tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
        ages = torch.hstack([ages, pad])

        # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
        tokens = tokens.masked_fill(ages > m, -1)
        ages = ages.masked_fill(ages > m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    if padding != 'nopad':
        # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
        tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # cut to maintain the block size
    #TODO it would be better to use the strategy defined by the "select" parameter
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    if padding != 'nopad':
        # if the first token is a "no event" token, mask it and the corresponding target
        x = x.masked_fill((x == 0) * (y == 1), 0)
        y = y.masked_fill(x == 0, 0)
        b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f


def print_data_statistics(sample, label, tab="   ", newline=True):
    """Print statistics for a tensor/array, handling both integer and float types."""
    print("data {} stats:".format(label))
    print("{}shape: {}".format(tab, sample.shape))
    print("{}dtype: {}".format(tab, sample.dtype))
    
    # Convert to numpy for consistent handling
    if torch.is_tensor(sample):
        sample_np = sample.detach().cpu().numpy()
        is_integer = sample.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, 
                                        torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    else:
        sample_np = np.asarray(sample)
        is_integer = np.issubdtype(sample_np.dtype, np.integer)
    
    # Basic stats that work for all types
    print("{}min: {}".format(tab, sample_np.min()))
    print("{}max: {}".format(tab, sample_np.max()))
    
    # Mean and std: convert to float if integer type
    if is_integer:
        sample_float = sample_np.astype(np.float64)
        print("{}mean: {:.4f}".format(tab, sample_float.mean()))
        print("{}std: {:.4f}".format(tab, sample_float.std()))
    else:
        print("{}mean: {:.4f}".format(tab, sample_np.mean()))
        print("{}std: {:.4f}".format(tab, sample_np.std()))
    
    # Percentiles: use numpy percentile which handles both types
    try:
        percentiles = np.percentile(sample_np, [0, 25, 50, 75, 100])
        print("{}percentiles: {}".format(tab, percentiles))
    except Exception as e:
        print("{}percentiles: <error: {}>".format(tab, str(e)))
    
    if newline:
        print()


def extract_icd_code(text):
    pattern = r'[A-Z]\d{2}(?=\s|$|[)\],.])' 
    match = re.search(pattern, text)
    return match.group(0) if match else None
