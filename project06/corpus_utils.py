from __future__ import annotations

import re
from typing import List

# fetch_corpus.py から流用: 日本語(ひらがな/カタカナ/漢字)を含むか判定
_RE_HAS_JA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

# 文末区切り。「。」で区切る(! ? も文末として扱う)
_RE_SPLIT = re.compile(r"(?<=[。！？])")


def clean_line(s: str) -> str:
    # fetch_corpus.py から流用: 前後空白除去 + 途中の改行・空白を詰める
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def split_sentences(text: str) -> List[str]:
    """「。」区切りで文に分割する。各文の末尾区切り文字は除いて返す。"""
    out: List[str] = []
    for chunk in _RE_SPLIT.split(text):
        s = clean_line(chunk)
        s = s.rstrip("。！？")  # 文末区切りは除く
        s = s.lstrip("、")  # 追加: 文頭の読点は除去(分割断片の先頭「、」対策)
        if s:
            out.append(s)
    return out


def is_valid_sentence(s: str, min_len: int, max_len: int) -> bool:
    if len(s) < min_len or len(s) > max_len:
        return False
    if not _RE_HAS_JA.search(s):
        return False
    return True
