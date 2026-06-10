from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import fugashi
import unidic

# UniDic の導入は pip + `python -m unidic download` が必要
_TAGGER = fugashi.Tagger(f'-d "{unidic.DICDIR}"')

# 用言(語尾を持つ自立語)
YOGEN_POS1 = {"動詞", "形容詞"}
# 追加: 名詞・形状詞(形容動詞)も起点にする(学生→です, 元気→だよね 等)
NOMINAL_POS1 = {"名詞", "形状詞"}
# 語尾としてマージする後続品詞(助動詞 + 補助の助詞「て/で」+ 補助動詞「いる/ある」等)
SUFFIX_POS1 = {"助動詞", "助詞", "動詞", "形容詞"}
# 追加: 語尾に含める助詞の pos2(接続助詞=て/ば/けど/から/ので/のに/が/し、終助詞=ね/よ/よね、準体助詞=の)
# 格助詞(全力「で」/家「に」)は除外することで名詞起点のノイズを防ぐ
SUFFIX_PARTICLE_POS2 = {"接続助詞", "終助詞", "準体助詞"}


def _safe_get(feat, name: str) -> str:
    v = getattr(feat, name, "")
    if v is None:
        return ""
    return str(v)


@dataclass
class StemSuffix:
    stem: str        # 語幹(辞書形の不変部。例: 寒い→寒, 目立つ→目立)
    suffix: str      # 語尾(活用語尾 + 後続助動詞/補助表現。例: かった, っている)
    pos1: str        # 用言の品詞(動詞/形容詞)
    start_char: int  # 原文中の用言開始位置(CTX 切り出し用)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _split_stem(surface: str, orth_base: str) -> Tuple[str, str]:
    """
    用言の表層形と辞書形から、語幹(不変部)と活用語尾を分割する。
      返り値: (stem, infl)  infl は活用語尾(後続助動詞は含まない)
    例: surface=寒かっ orth_base=寒い -> stem=寒, infl=かっ
        surface=目立っ orth_base=目立つ -> stem=目立, infl=っ
        surface=来      orth_base=来る   -> stem=来, infl=""
    共通接頭辞が空(サ変「し」/「する」等)の場合は表層形そのものを stem とする。
    """
    cpl = _common_prefix_len(surface, orth_base)
    if cpl == 0:
        # 不規則(する/来る の一部)など: 表層形を語幹扱いにしてフォールバック
        return surface, ""
    return surface[:cpl], surface[cpl:]


def extract_stem_suffix(
    text: str,
    allow_pos1: Optional[set] = None,
) -> List[StemSuffix]:
    """
    原文から用言・名詞を起点に (stem, suffix) を抽出する。
      - 用言(動詞/形容詞)を見つけたら、その活用語尾以降から、
        後続の助動詞・補助の助詞「て/で/ので/から/けど 等」・補助動詞「いる/ある」・終助詞を語尾としてまとめる。
      - 追加: 名詞/形状詞も起点にする(stem=表層)。ただし後続に断定助動詞や助詞が続く場合のみ採用(単独名詞のノイズ回避)。
    """
    if allow_pos1 is None:
        allow_pos1 = YOGEN_POS1 | NOMINAL_POS1  # 追加: 名詞・形状詞も起点に

    toks = list(_TAGGER(text))

    # 原文中の各 surface 開始位置を求める(make_rerank と同じ追従方式)
    starts: List[int] = []
    cursor = 0
    for w in toks:
        idx = text.find(w.surface, cursor)
        if idx < 0:
            idx = cursor
        starts.append(idx)
        cursor = idx + len(w.surface)

    out: List[StemSuffix] = []
    i = 0
    n = len(toks)
    while i < n:
        w = toks[i]
        pos1 = _safe_get(w.feature, "pos1")
        if pos1 not in allow_pos1:
            i += 1
            continue

        is_nominal = pos1 in NOMINAL_POS1  # 追加: 名詞・形状詞起点フラグ
        if is_nominal:
            # 名詞/形状詞は活用しないので表層そのものを stem とし、活用語尾は空
            stem, infl = w.surface, ""
        else:
            orth_base = _safe_get(w.feature, "orthBase")
            if not orth_base or orth_base == "*":
                orth_base = w.surface
            stem, infl = _split_stem(w.surface, orth_base)
        suffix = infl

        # 後続の語尾要素をマージ(助動詞・接続/補助の助詞・補助動詞いる/ある/くる/しまう・終助詞 等)
        j = i + 1
        while j < n:
            t = toks[j]
            tpos1 = _safe_get(t.feature, "pos1")
            tpos2 = _safe_get(t.feature, "pos2")
            if tpos1 == "助動詞":
                suffix += t.surface
                j += 1
                continue
            # 接続助詞(て/ば/けど/から/ので/のに/が/し)・終助詞(ね/よ/よね)・準体助詞(の)を語尾に含める
            # 格助詞(全力「で」/家「に」)は対象外なので名詞起点のノイズを防げる
            if tpos1 == "助詞" and tpos2 in SUFFIX_PARTICLE_POS2:
                suffix += t.surface
                j += 1
                continue
            # 補助動詞/補助形容詞(いる/ある/くる/しまう/ない 等)
            # 名詞起点では後続の自立用言は別文節なのでマージしない(「昨日」+「来た」等のノイズ回避)
            if not is_nominal and tpos1 in ("動詞", "形容詞"):
                suffix += t.surface
                j += 1
                continue
            # 形状詞の助動詞語幹(そう/よう 等。例: 寒そう)を語尾に含める
            if tpos1 == "形状詞" and _safe_get(t.feature, "pos2") == "助動詞語幹":
                suffix += t.surface
                j += 1
                continue
            break

        # 用言起点: 語尾が空(辞書形そのまま等)はスキップ。stem が空もスキップ
        # 名詞起点: 後続(suffix)が無いと単なる名詞なのでスキップ(ノイズ回避)
        if stem and suffix:
            out.append(StemSuffix(stem=stem, suffix=suffix, pos1=pos1, start_char=starts[i]))
        i = j if j > i + 1 else i + 1

    return out


if __name__ == "__main__":
    samples = [
        "寒かったので家にいた",
        "それは目立っている建物だ",
        "全力で走りました",
        "あまり美しくない景色",
        "弟に食べさせられた",
        "昨日来た人",
        "宿題をしました",
        "明日は寒そうだ",
        # 追加: 接続/丁寧/終助詞/名詞起点の新パターン
        "私は学生です",
        "彼は元気だよね",
        "今日は雨だから",
        "寒いので家にいる",
        "明日は行くから",
        "全力で走りますけど",
    ]
    for s in samples:
        print("===", s)
        for r in extract_stem_suffix(s):
            print(f"  stem={r.stem} | suffix={r.suffix} | pos1={r.pos1} | start={r.start_char}")
