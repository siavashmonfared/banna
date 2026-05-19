"""GAIA dataset loader.

GAIA ("General AI Assistants", Mialon et al. 2023) is a 465-question
benchmark split into 3 levels. The *validation* set (165 questions) is
publicly available on HuggingFace and is what we grade against; the
test set (300 questions) is held out behind a leaderboard.

Dataset shape on HF (`gaia-benchmark/GAIA`, config `2023_all` for
validation):
  - task_id           str
  - Question          str
  - Level             int (1, 2, or 3)
  - Final answer      str  (the exact-match target)
  - file_name         str (optional attachment, empty if none)
  - file_path         str (resolved locally when loaded)
  - Annotator Metadata dict (steps, tools, reasoning — we ignore)

We return a `GAIATask` dataclass per row so the runner sees a stable
shape regardless of HF column renames.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "gaia-benchmark/GAIA"
DEFAULT_SPLIT = "validation"
DEFAULT_CONFIG = "2023_all"


@dataclass
class GAIATask:
    task_id: str
    question: str
    level: int
    answer: str              # ground-truth, exact-match target
    file_name: str = ""
    file_path: str = ""      # absolute path when file was downloaded
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_attachment(self) -> bool:
        return bool(self.file_name)


def load_gaia(
    *,
    split: str = DEFAULT_SPLIT,
    config: str = DEFAULT_CONFIG,
    levels: Iterable[int] | None = None,
    limit: int | None = None,
    dataset_name: str = DEFAULT_DATASET,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> list[GAIATask]:
    """Load GAIA tasks from HuggingFace.

    Parameters
    ----------
    split       : "validation" (public) or "test" (held-out)
    config      : "2023_all", "2023_level1", "2023_level2", or "2023_level3"
    levels      : optional filter; e.g. `{1}` for only L1
    limit       : optional cap on returned tasks
    cache_dir   : HF cache directory (defaults to ~/.cache/huggingface)
    hf_token    : override for HF_TOKEN env var
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "`datasets` not installed. `pip install datasets` to load GAIA."
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        dataset_name,
        config,
        split=split,
        cache_dir=cache_dir,
        token=token,
    )

    tasks: list[GAIATask] = []
    for row in ds:
        lvl = int(row.get("Level", row.get("level", 0)) or 0)
        if levels is not None and lvl not in set(levels):
            continue
        file_name = str(row.get("file_name") or "")
        file_path = str(row.get("file_path") or "")
        # The HF row gives `file_path` as a repo-relative pointer
        # ("2023/validation/<id>.docx"), not a local file. Materialize the
        # attachment via the Hub file API so `read_file` / pdf / xlsx tools
        # can open it. Failures here are non-fatal: we keep the relative
        # path so the agent at least sees that an attachment was intended.
        if file_name:
            resolved = _download_attachment(
                file_path or f"2023/{split}/{file_name}",
                dataset_name=dataset_name,
                token=token,
                cache_dir=cache_dir,
            )
            if resolved:
                file_path = resolved
        task = GAIATask(
            task_id=str(row.get("task_id") or row.get("id") or ""),
            question=str(row.get("Question") or row.get("question") or ""),
            level=lvl,
            answer=str(row.get("Final answer") or row.get("final_answer") or ""),
            file_name=file_name,
            file_path=file_path,
            metadata=dict(row.get("Annotator Metadata") or {}),
        )
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _download_attachment(
    repo_relative_path: str,
    *,
    dataset_name: str,
    token: str | None,
    cache_dir: str | None,
) -> str:
    """Fetch a GAIA attachment from the HF Hub and return its absolute
    local path. Returns "" on any failure."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return ""
    try:
        return hf_hub_download(
            repo_id=dataset_name,
            filename=repo_relative_path,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    except Exception:
        return ""


def load_gaia_from_jsonl(path: str | Path) -> list[GAIATask]:
    """Load GAIA tasks from a local JSONL file — useful when the HF dataset
    is unavailable or when running against a custom subset.

    Expected fields per line: task_id, question, level, answer, file_name?,
    file_path?, metadata?.
    """
    import json

    p = Path(path)
    tasks: list[GAIATask] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tasks.append(GAIATask(
            task_id=str(row.get("task_id", "")),
            question=str(row.get("question", "")),
            level=int(row.get("level", 0)),
            answer=str(row.get("answer", "")),
            file_name=str(row.get("file_name", "")),
            file_path=str(row.get("file_path", "")),
            metadata=dict(row.get("metadata") or {}),
        ))
    return tasks
