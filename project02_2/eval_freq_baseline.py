#!/usr/bin/env python3
"""頻度ベースライン診断: eval.jsonl.gz で「母音列の頻度1位を選ぶだけ」で何%当たるかを測る。
   追加: --lora_dir を指定すると LoRA 推論の正解率も同じデータで測り、top1 と比較する。"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict


def load_vowel_rank(vowel2cands_path: str):
    # word -> その母音列内での頻度順位(0が最頻) を引けるようにする
    with open(vowel2cands_path, "r", encoding="utf-8") as f:
        v2 = json.load(f)
    rank = {}
    for vowels, cands in v2.items():
        rank[vowels] = {w: i for i, w in enumerate(cands)}
    return rank


# 追加: infer.py と一致させたプロンプト(ANSWER:\n まで)。推論スコアの土俵を揃える
def build_prompt(ctx: str, vowels: str, cands) -> str:
    lines = []
    lines.append("あなたは日本語IMEの予測器です。")
    lines.append("出力は候補番号のみ。")
    lines.append(f"CTX: {ctx}")
    lines.append(f"VOWELS: {vowels}")
    lines.append("CANDIDATES:")
    for i, w in enumerate(cands, start=1):
        lines.append(f"{i}) {w}")
    lines.append("ANSWER:\n")
    return "\n".join(lines)


# 追加: infer.py の score_candidate と同じ。prompt + 候補番号文字列 の対数尤度合算
def score_candidate(model, tok, prompt_ids, num_str: str, torch) -> float:
    cont_ids = tok(num_str, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf")
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([cont_ids], device=prompt_ids.device)], dim=1
    )
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    base = prompt_ids.shape[1] - 1
    total = 0.0
    for k, tid in enumerate(cont_ids):
        total += log_probs[base + k, tid].item()
    return total


# 追加: LoRA モデルを 1 サンプル推論して 1-based の予測候補番号を返す
def predict_index(model, tok, ex, torch) -> int:
    prompt = build_prompt(ex["ctx"], ex["vowels"], ex["cands"])
    prompt_ids = tok(prompt, return_tensors="pt").to(model.device)["input_ids"]
    best_i, best_s = 1, float("-inf")
    for i in range(1, len(ex["cands"]) + 1):
        s = score_candidate(model, tok, prompt_ids, str(i), torch)
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", default="../dataset/project02_2/eval.jsonl.gz")
    ap.add_argument("--vowel2cands", default="vowel2cands.json")
    ap.add_argument("--max_records", type=int, default=20000)  # 0 なら全件使用
    # 追加: LoRA 推論で比較する場合の引数(指定なしなら頻度ベースラインのみ)
    ap.add_argument("--lora_dir", default=None, help="指定すると LoRA 推論の正解率も測る")
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--revision", default="2aaa97c53d")
    # 追加: 推論した各サンプル(ctx/正解/モデル予測)を表示する。--vis_n で表示件数
    ap.add_argument("--vis", action="store_true", default=False, help="推論結果を1件ずつ表示する")
    ap.add_argument("--vis_n", type=int, default=50, help="--vis 時に表示する最大件数")
    args = ap.parse_args()

    rank = load_vowel_rank(args.vowel2cands)

    # 追加: LoRA 指定時のみモデルをロード(torch 等は推論時だけ import)
    model = tok = torch = None
    if args.lora_dir:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model:", args.base_model, "lora:", args.lora_dir)
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, revision=args.revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            revision=args.revision,
        )
        model = PeftModel.from_pretrained(base, args.lora_dir)
        model.eval()
        print("Ready.")

    # 追加: eval 全体の件数を先に数える(20000 が全体の何%かを出すため)
    eval_total = 0
    with gzip.open(args.eval_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_total += 1

    total = 0
    freq_correct = 0            # 頻度1位ベースラインの正解数
    lora_correct = 0            # 追加: LoRA 推論の正解数
    vis_rows = []               # 追加: --vis 表示用の推論結果
    # 同じ母音列で正解語が何種類に割れているか
    vowel_golds = defaultdict(set)

    with gzip.open(args.eval_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            cands = ex["cands"]
            vowels = ex["vowels"]
            ans_idx = int(ex["answer"])  # 1-based
            gold = cands[ans_idx - 1]

            r = rank.get(vowels, {})
            # 候補の中で「母音列頻度が最も高い(順位が最小)」語 = 文脈無視ベースラインの選択
            best = min(cands, key=lambda w: r.get(w, 10**9))

            total += 1
            if best == gold:
                freq_correct += 1

            # 追加: LoRA 推論(指定時のみ)。同じサンプルで予測番号が正解番号と一致するか
            if model is not None:
                pred_idx = predict_index(model, tok, ex, torch)
                if pred_idx == ans_idx:
                    lora_correct += 1
                # 追加: --vis 用に推論したサンプルを記録(上限 vis_n まで)
                if args.vis and len(vis_rows) < args.vis_n:
                    vis_rows.append((ex, gold, best, cands[pred_idx - 1]))

            vowel_golds[vowels].add(gold)

            # 変更: max_records=0 なら全件使用(上限なし)
            if args.max_records > 0 and total >= args.max_records:
                break

    print("=" * 60)
    print(f"eval total (全件)     : {eval_total}")  # 追加: eval データ全体の件数
    print(f"used records (集計対象): {total}  ({total/eval_total*100:.2f}% of 全件)")  # 追加: 集計に使った件数と全体比
    print(f"freq-top1 baseline acc: {freq_correct/total*100:.2f}%  ({freq_correct}/{total})")
    print(f"  -> この数字が高いほど『文脈を見なくても頻度だけで当たる』甘い評価データ")
    # 追加: LoRA 推論の正解率と、頻度ベースラインとの差分(文脈を使えてるかの指標)
    if model is not None:
        print(f"LoRA infer acc        : {lora_correct/total*100:.2f}%  ({lora_correct}/{total})")
        print(f"  -> diff (LoRA - top1) : {(lora_correct-freq_correct)/total*100:+.2f}pt  (プラスなら文脈を活用できている)")
    # 同じ母音列で正解が2種類以上に割れている割合(文脈依存度の目安)
    split = sum(1 for v, gs in vowel_golds.items() if len(gs) >= 2)
    print(f"vowel patterns        : {len(vowel_golds)}  (うち正解が2種類以上に割れ: {split})")
    print("=" * 60)

    # 追加: --vis 指定時、推論した各サンプルを表示(正解/頻度1位/モデル予測 と ○×)
    if args.vis and vis_rows:
        print(f"\n--- 推論結果 {len(vis_rows)} 件 ---")
        for k, (ex, gold, best, pred) in enumerate(vis_rows, 1):
            mark = "○" if pred == gold else "×"
            print(f"[{k}] CTX={ex['ctx']!r}  VOWELS=[{ex['vowels']}]")
            print(f"     正解={gold!r}  / 頻度1位={best!r}  / モデル予測={pred!r}  -> {mark}")


if __name__ == "__main__":
    main()
