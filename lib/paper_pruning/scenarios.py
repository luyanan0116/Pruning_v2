from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def load_scenario_manifest(
    path: str | Path,
    tokenizer,
    max_length: int,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load task/language/prompt/input scenarios from JSON or JSONL.

    Each record may contain ``text`` or ``input_ids`` and optional metadata:
    ``base_sample_id``, ``scenario_id``, ``task``, ``language``,
    ``prompt_template`` and arbitrary ``metadata``. The collector combines this
    scenario identity with context-length and crop scenarios.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(source.read_text(encoding="utf-8"))
        records = loaded["records"] if isinstance(loaded, dict) and "records" in loaded else loaded
    if not isinstance(records, list):
        raise ValueError("scenario manifest must contain a list of records")

    result: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if limit is not None and len(result) >= limit:
            break
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {index} must be an object")
        if "input_ids" in record:
            ids = torch.as_tensor(record["input_ids"], dtype=torch.long).reshape(1, -1)
        elif "text" in record:
            encoded = tokenizer(
                str(record["text"]),
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            ids = encoded.input_ids
        else:
            raise ValueError(f"manifest record {index} needs text or input_ids")
        if ids.shape[1] < 8:
            continue
        metadata = dict(record.get("metadata", {}))
        for key in ("task", "language", "prompt_template", "domain"):
            if key in record:
                metadata[key] = record[key]
        result.append(
            {
                "input_ids": ids[:, :max_length],
                "base_sample_id": record.get("base_sample_id", index),
                "scenario_id": record.get("scenario_id", metadata or "manifest"),
                "metadata": metadata,
            }
        )
    if not result:
        raise ValueError("scenario manifest produced no sequence with at least 8 tokens")
    return result
