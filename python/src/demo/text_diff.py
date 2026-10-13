"""Line-by-line comparison of two printed programs, marking what was added, removed or changed and where within a line."""
import difflib
import re
from itertools import accumulate
from typing import Dict, List, Sequence, Tuple

_TOKEN = re.compile(r"[A-Za-z_][\w#]*|\d+(?:\.\d+)?|\s+|.")
_MODIFIED_RATIO = 0.5  # lines less alike than this are shown as one removed and one added, not as a modification


def line(text: str, kind: str, spans: Sequence[Tuple[int, int]] = ()) -> Dict:
    return {"text": text, "kind": kind, "spans": [list(span) for span in spans]}


def _offsets(tokens: Sequence[str]) -> List[int]:
    return [0, *accumulate(len(token) for token in tokens)]


def _changed_spans(old: str, new: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Character ranges that differ between two lines, for the old and the new text."""
    old_tokens, new_tokens = _TOKEN.findall(old), _TOKEN.findall(new)
    old_at, new_at = _offsets(old_tokens), _offsets(new_tokens)
    old_spans, new_spans = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            old_spans.append((old_at[i1], old_at[i2]))
        if j2 > j1:
            new_spans.append((new_at[j1], new_at[j2]))
    return old_spans, new_spans


def diff_lines(before: List[str], after: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Mark each line of `before` and `after` as same, added, removed or changed (with the changed ranges)."""
    old: List[Dict] = [{}] * len(before)
    new: List[Dict] = [{}] * len(after)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                old[i], new[j] = line(before[i], "same"), line(after[j], "same")
            continue
        paired = 0
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                i, j = i1 + k, j1 + k
                if difflib.SequenceMatcher(None, before[i], after[j]).ratio() < _MODIFIED_RATIO:
                    break
                old_spans, new_spans = _changed_spans(before[i], after[j])
                old[i], new[j] = line(before[i], "changed", old_spans), line(after[j], "changed", new_spans)
                paired += 1
        for i in range(i1 + paired, i2):
            old[i] = line(before[i], "removed")
        for j in range(j1 + paired, j2):
            new[j] = line(after[j], "added")
    return old, new
