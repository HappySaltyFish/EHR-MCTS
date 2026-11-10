"""Shared RAG document budget and prompt rendering for PRM training and inference."""

from typing import Sequence

from src.sample.formatters import build_user_content


PROMPT_RESERVE_TOKENS = 1024


def build_prm_prompt(
    tokenizer,
    system_prompt: str,
    question_text: str,
    solution_text: str,
    max_token_len: int,
    related_docs: Sequence[str] = (),
    step_tag: str = "ки",
) -> str:
    """Keep whole documents within the shared budget and full prompt limit."""
    def render(docs: Sequence[str]) -> str:
        doc_block = "".join(f"Document {i + 1}: {doc}\n\n" for i, doc in enumerate(docs))
        content = build_user_content(doc_block, question_text, solution_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return raw.replace(f" {step_tag}\n", f" {step_tag}")

    kept = []
    used = 0
    raw = render(kept)
    doc_budget = max(0, max_token_len - PROMPT_RESERVE_TOKENS)
    for doc in related_docs:
        cost = len(tokenizer(doc, add_special_tokens=False)["input_ids"]) + 1
        if used + cost > doc_budget:
            break
        candidate = render([*kept, doc])
        # The model input appends one pad token after tokenization.
        if len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) + 1 > max_token_len:
            break
        kept.append(doc)
        used += cost
        raw = candidate
    return raw
