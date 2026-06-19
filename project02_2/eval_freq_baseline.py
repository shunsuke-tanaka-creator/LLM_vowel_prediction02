#!/usr/bin/env python3
"""頻度ベースライン診断: eval.jsonl.gz で「母音列の頻度1位を選ぶだけ」で何%当たるかを測る。
   追加: --lora_dir を指定すると LoRA 推論の正解率も同じデータで測り、top1 と比較する。"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict

from tqdm import tqdm  # 追加: 推論進捗バー表示用


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


# 変更: LoRA モデルを 1 サンプル推論し、スコア降順の候補番号(1-based)リストを返す(top-k 比較用)
def predict_ranking(model, tok, ex, torch):
    prompt = build_prompt(ex["ctx"], ex["vowels"], ex["cands"])
    prompt_ids = tok(prompt, return_tensors="pt").to(model.device)["input_ids"]
    scored = []
    for i in range(1, len(ex["cands"]) + 1):
        s = score_candidate(model, tok, prompt_ids, str(i), torch)
        scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for (i, _) in scored]


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
    freq_top3 = 0               # 追加: 頻度上位3に正解が入る数
    lora_correct = 0            # 追加: LoRA 推論の正解数
    lora_top3 = 0               # 追加: LoRA 上位3に正解が入る数
    vis_rows = []               # 追加: --vis 表示用の推論結果
    # 同じ母音列で正解語が何種類に割れているか
    vowel_golds = defaultdict(set)

    # 追加: 進捗バー用。処理予定の総件数(上限ありならその値、なければ全件)
    plan_total = args.max_records if args.max_records > 0 else eval_total

    with gzip.open(args.eval_path, "rt", encoding="utf-8") as f:
        # 変更: tqdm で推論進捗を表示(total=処理予定件数)
        for line in tqdm(f, total=plan_total, desc="infer" if model is not None else "scan"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            cands = ex["cands"]
            vowels = ex["vowels"]
            ans_idx = int(ex["answer"])  # 1-based
            gold = cands[ans_idx - 1]

            r = rank.get(vowels, {})
            # 候補を母音列頻度の高い順(順位が小さい順)に並べる = 文脈無視ベースラインのランキング
            freq_rank = sorted(cands, key=lambda w: r.get(w, 10**9))
            best = freq_rank[0]  # 頻度1位

            total += 1
            if best == gold:
                freq_correct += 1
            if gold in freq_rank[:3]:  # 追加: 頻度 top3
                freq_top3 += 1

            # 追加: LoRA 推論(指定時のみ)。スコア降順ランキングで top1/top3 を判定
            if model is not None:
                ranking = predict_ranking(model, tok, ex, torch)  # 1-based 候補番号の降順
                if ranking[0] == ans_idx:
                    lora_correct += 1
                if ans_idx in ranking[:3]:  # 追加: LoRA top3
                    lora_top3 += 1
                # 追加: --vis 用に推論したサンプルを記録(上限 vis_n まで)
                if args.vis and len(vis_rows) < args.vis_n:
                    pred_top3 = [cands[i - 1] for i in ranking[:3]]  # 予測 top3 の単語
                    vis_rows.append((ex, gold, best, pred_top3))

            vowel_golds[vowels].add(gold)

            # 変更: max_records=0 なら全件使用(上限なし)
            if args.max_records > 0 and total >= args.max_records:
                break

    print("=" * 60)
    print(f"eval total (全件)     : {eval_total}")  # 追加: eval データ全体の件数
    print(f"used records (集計対象): {total}  ({total/eval_total*100:.2f}% of 全件)")  # 追加: 集計に使った件数と全体比
    print(f"freq-top1 baseline acc: {freq_correct/total*100:.2f}%  ({freq_correct}/{total})")
    print(f"freq-top3 baseline acc: {freq_top3/total*100:.2f}%  ({freq_top3}/{total})")  # 追加: 頻度 top3
    print(f"  -> この数字が高いほど『文脈を見なくても頻度だけで当たる』甘い評価データ")
    # 追加: LoRA 推論の正解率(top1/top3)と、頻度ベースラインとの差分(文脈を使えてるかの指標)
    if model is not None:
        print(f"LoRA infer top1 acc   : {lora_correct/total*100:.2f}%  ({lora_correct}/{total})")
        print(f"LoRA infer top3 acc   : {lora_top3/total*100:.2f}%  ({lora_top3}/{total})")  # 追加: LoRA top3
        print(f"  -> diff top1 (LoRA-freq): {(lora_correct-freq_correct)/total*100:+.2f}pt")
        print(f"  -> diff top3 (LoRA-freq): {(lora_top3-freq_top3)/total*100:+.2f}pt  (プラスなら文脈を活用できている)")
    # 同じ母音列で正解が2種類以上に割れている割合(文脈依存度の目安)
    split = sum(1 for v, gs in vowel_golds.items() if len(gs) >= 2)
    print(f"vowel patterns        : {len(vowel_golds)}  (うち正解が2種類以上に割れ: {split})")
    print("=" * 60)

    # 追加: --vis 指定時、推論した各サンプルを表示(正解/頻度1位/モデル予測top3 と ○×)
    if args.vis and vis_rows:
        print(f"\n--- 推論結果 {len(vis_rows)} 件 ---")
        for k, (ex, gold, best, pred_top3) in enumerate(vis_rows, 1):
            mark = "○" if gold in pred_top3 else "×"  # top3 に正解が入っていれば○
            print(f"[{k}] CTX={ex['ctx']!r}  VOWELS=[{ex['vowels']}]")
            print(f"     正解={gold!r}  / 頻度1位={best!r}  / モデル予測top3={pred_top3}  -> {mark}")


if __name__ == "__main__":
    main()
