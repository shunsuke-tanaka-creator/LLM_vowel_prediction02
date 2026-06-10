from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import fugashi
import unidic

# UniDic の導入は pip + `python -m unidic download` が必要 :contentReference[oaicite:7]{index=7}
_TAGGER = fugashi.Tagger(f'-d "{unidic.DICDIR}"')


@dataclass
class MorphToken:
    surface: str
    pos1: str
    pos2: str
    lemma: str
    orth_base: str
    kana_base: str  # 読み（カタカナが多い）
    kana: str  # 追加: 活用形の読み（カタカナ）


def _safe_get(feat, name: str) -> str:
    v = getattr(feat, name, "")
    if v is None:
        return ""
    return str(v)


def tokenize(text: str) -> List[MorphToken]:
    out: List[MorphToken] = []
    for w in _TAGGER(text):
        feat = w.feature
        pos1 = _safe_get(feat, "pos1")
        pos2 = _safe_get(feat, "pos2")
        lemma = _safe_get(feat, "lemma")
        orth_base = _safe_get(feat, "orthBase")
        kana_base = _safe_get(feat, "kanaBase")
        kana = _safe_get(feat, "kana")  # 追加: 活用形の読み

        # orth_base が '*' っぽい時のフォールバック
        if not orth_base or orth_base == "*":
            orth_base = lemma if lemma and lemma != "*" else w.surface

        # kana_base が無い時は lForm / kana を試す（UniDicのフィールド説明あり :contentReference[oaicite:8]{index=8}）
        if not kana_base or kana_base == "*":
            kana_base = _safe_get(feat, "lForm")
        if not kana_base or kana_base == "*":
            kana_base = _safe_get(feat, "kana")

        # 追加: kana(活用形読み)が無い時は kana_base / surface でフォールバック
        if not kana or kana == "*":
            kana = kana_base if kana_base and kana_base != "*" else w.surface

        out.append(MorphToken(
            surface=w.surface,
            pos1=pos1,
            pos2=pos2,
            lemma=lemma,
            orth_base=orth_base,
            kana_base=kana_base,
            kana=kana,  # 追加
        ))
    return out


def extract_content_words(
    text: str,
    allow_pos1: Optional[set[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    content words 列を返す：
      [(word, pos1, reading), ...]

    追加: 活用形(surface)と活用形の読み(kana)を返す。
    追加: 自立語に連続する助動詞を活用語尾として直前の自立語にマージする
          （例: 寒かっ+た → 寒かった / サムカッ+タ → サムカッタ）。
    """
    if allow_pos1 is None:
        allow_pos1 = {"名詞", "形容詞", "動詞", "感動詞"}  # 定型句枠として感動詞を入れる（不要なら外す）

    toks = tokenize(text)
    out: List[Tuple[str, str, str]] = []
    cur_word: Optional[str] = None
    cur_pos1: str = ""
    cur_reading: str = ""
    for t in toks:
        if t.pos1 in allow_pos1:
            # 追加: 直前の自立語を確定してから新しい自立語を開始
            if cur_word is not None:
                out.append((cur_word, cur_pos1, cur_reading))
            cur_word = t.surface
            cur_pos1 = t.pos1
            cur_reading = t.kana  # 追加: 活用形の読み
        elif t.pos1 == "助動詞" and cur_word is not None:
            # 追加: 自立語に連続する助動詞は活用語尾としてマージ
            cur_word += t.surface
            cur_reading += t.kana
        else:
            # 追加: 助詞・記号などが来たら現在の自立語を確定し連結を打ち切る
            if cur_word is not None:
                out.append((cur_word, cur_pos1, cur_reading))
                cur_word = None
    if cur_word is not None:
        out.append((cur_word, cur_pos1, cur_reading))
    return out


def extract_content_words_with_pos(
    text: str,
    allow_pos1: Optional[set[str]] = None,
) -> List[Tuple[str, str, str, int]]:
    """
    追加: extract_content_words と同じまとまり単位で、各 word の
    原文中の開始文字位置(start_char)も返す。
      [(word, pos1, reading, start_char), ...]
    CTX を原文の部分文字列そのままにするために使う。
    """
    if allow_pos1 is None:
        allow_pos1 = {"名詞", "形容詞", "動詞", "感動詞"}

    out: List[Tuple[str, str, str, int]] = []
    cur_word: Optional[str] = None
    cur_pos1: str = ""
    cur_reading: str = ""
    cur_start: int = 0
    cursor = 0  # 追加: 原文中の現在位置(surface を順に消費)
    for w in _TAGGER(text):
        feat = w.feature
        pos1 = _safe_get(feat, "pos1")
        kana = _safe_get(feat, "kana")
        kana_base = _safe_get(feat, "kanaBase")
        if not kana or kana == "*":
            kana = kana_base if kana_base and kana_base != "*" else w.surface

        # 追加: 原文中での surface 開始位置を求める(空白等で飛ぶ場合に追従)
        idx = text.find(w.surface, cursor)
        if idx < 0:
            idx = cursor
        tok_start = idx
        cursor = idx + len(w.surface)

        if pos1 in allow_pos1:
            if cur_word is not None:
                out.append((cur_word, cur_pos1, cur_reading, cur_start))
            cur_word = w.surface
            cur_pos1 = pos1
            cur_reading = kana
            cur_start = tok_start
        elif pos1 == "助動詞" and cur_word is not None:
            cur_word += w.surface
            cur_reading += kana
        else:
            if cur_word is not None:
                out.append((cur_word, cur_pos1, cur_reading, cur_start))
                cur_word = None
    if cur_word is not None:
        out.append((cur_word, cur_pos1, cur_reading, cur_start))
    return out
