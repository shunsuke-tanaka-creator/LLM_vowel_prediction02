from __future__ import annotations
from pykakasi import kakasi

_kks = kakasi()

import re
from typing import List

import jaconv

# =========================
# ひらがな → 母音（a,i,u,e,o,n）
# =========================

BASE_VOWEL = {
    # あ行
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",

    # か行
    "か": "a", "き": "i", "く": "u", "け": "e", "こ": "o",
    "が": "a", "ぎ": "i", "ぐ": "u", "げ": "e", "ご": "o",

    # さ行
    "さ": "a", "し": "i", "す": "u", "せ": "e", "そ": "o",
    "ざ": "a", "じ": "i", "ず": "u", "ぜ": "e", "ぞ": "o",

    # た行
    "た": "a", "ち": "i", "つ": "u", "て": "e", "と": "o",
    "だ": "a", "ぢ": "i", "づ": "u", "で": "e", "ど": "o",

    # な行
    "な": "a", "に": "i", "ぬ": "u", "ね": "e", "の": "o",

    # は行
    "は": "a", "ひ": "i", "ふ": "u", "へ": "e", "ほ": "o",
    "ば": "a", "び": "i", "ぶ": "u", "べ": "e", "ぼ": "o",
    "ぱ": "a", "ぴ": "i", "ぷ": "u", "ぺ": "e", "ぽ": "o",

    # ま行
    "ま": "a", "み": "i", "む": "u", "め": "e", "も": "o",

    # や行
    "や": "a", "ゆ": "u", "よ": "o",

    # ら行
    "ら": "a", "り": "i", "る": "u", "れ": "e", "ろ": "o",

    # わ行
    "わ": "a", "ゐ": "i", "ゑ": "e", "を": "o",

    # 外来
    "ゔ": "u",
}

YOON_MAP = {
    "きゃ": "a", "きゅ": "u", "きょ": "o",
    "しゃ": "a", "しゅ": "u", "しょ": "o",
    "ちゃ": "a", "ちゅ": "u", "ちょ": "o",
    "にゃ": "a", "にゅ": "u", "にょ": "o",
    "ひゃ": "a", "ひゅ": "u", "ひょ": "o",
    "みゃ": "a", "みゅ": "u", "みょ": "o",
    "りゃ": "a", "りゅ": "u", "りょ": "o",
    "ぎゃ": "a", "ぎゅ": "u", "ぎょ": "o",
    "じゃ": "a", "じゅ": "u", "じょ": "o",
    "びゃ": "a", "びゅ": "u", "びょ": "o",
    "ぴゃ": "a", "ぴゅ": "u", "ぴょ": "o",
}

IGNORE_CHARS = set("っーゎゕゖ")

_RE_ALLOWED_WORD = re.compile(r"^[\u3040-\u309F\u4E00-\u9FFF々]+$")


def is_allowed_output_word(word: str) -> bool:
    return bool(_RE_ALLOWED_WORD.match(word))


def hira_to_vowels_strict(hira: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(hira):
        pair = hira[i:i + 2]
        if pair in YOON_MAP:
            out.append(YOON_MAP[pair])
            i += 2
            continue

        ch = hira[i]

        if ch == "ん":
            out.append("n")
            i += 1
            continue

        if ch in IGNORE_CHARS:
            i += 1
            continue

        v = BASE_VOWEL.get(ch)
        if v is not None:
            out.append(v)

        i += 1

    return out


def reading_to_vowel_str(reading: str) -> str:
    if not reading:
        return ""
    hira = jaconv.kata2hira(reading)
    vs = hira_to_vowels_strict(hira)
    return " ".join(vs)



def text_to_vowel_str(text: str) -> str:
    result = _kks.convert(text)  # ← 新API
    out = []
    for token in result:
        hira = token["hira"]
        out.extend(hira_to_vowels_strict(hira))
    return " ".join(out)


if __name__ == "__main__":
    samples = ["寒い", "暑い", "今日", "冬", "走る", "美しい", "ありがとう"]
    for s in samples:
        print(s, "->", text_to_vowel_str(s))
