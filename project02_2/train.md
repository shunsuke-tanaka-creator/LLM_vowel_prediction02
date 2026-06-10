# project02_2 学習手順 (LoRA リランカー)

母音列(a/i/u/e/o/n)+文脈 から、候補語リストの正解番号を当てる **リランカー** を MiniCPM4-0.5B + LoRA で学習する。
学習スクリプト本体は `train_rerank_lora.py`(`tanakamemo.txt` の実行例で参照)。

## 1. 全体パイプライン

```
build_vocab_from_cc100.py      # cc100-ja から語彙 + vowel2cands.json を作成
        ↓
make_rerank_dataset_cc100.py   # 学習/評価用 jsonl.gz を作成
        ↓
train_rerank_lora.py           # LoRA 学習 → lora_out_minicpm4/
        ↓
infer.py / serve_api.py        # 推論
```

## 2. データ形式

`make_rerank_dataset_cc100.py` が出力する 1 行 = 1 サンプル。

```json
{"ctx": "午後から", "vowels": "a e", "cands": ["風邪", "金", ...], "answer": "10"}
```

- `ctx`: 直前 `ctx_len`(既定4) 語ぶんの原文部分文字列。無ければ `<NONE>`。
- `vowels`: 正解語の母音列(空白区切り)。
- `cands`: `num_cands`(既定32) 個の候補。正解語(gold)を必ず含み、順序はシャッフル済み。
- `answer`: 正解候補の 1 始まり番号(文字列)。

## 3. プロンプト形式 (infer.py / serve_api.py と一致させること)

```
あなたは日本語IMEの予測器です。
出力は候補番号のみ。
CTX: {ctx}
VOWELS: {vowels}
CANDIDATES:
1) {cand1}
2) {cand2}
...
ANSWER:
{answer}
```

学習時は `ANSWER:\n` までを prompt、`{answer}` を続きとして与え、
**続き部分(answer トークン)だけに loss をかける**(prompt 部分は `label=-100` でマスク)。

## 4. ベースモデルと LoRA 設定 (adapter_config.json と一致)

- base: `openbmb/MiniCPM4-0.5B` (`trust_remote_code=True`, `revision=2aaa97c53d`)
- dtype: bfloat16
- LoRA: `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `bias=none`, `task_type=CAUSAL_LM`
- target_modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- tokenizer: `pad_token` が無ければ `eos_token` を代入

## 5. 学習コマンド

```bash
python -u train_rerank_lora.py \
  --train_path ../dataset/project02_2/train.jsonl.gz \
  --eval_path ../dataset/project02_2/eval.jsonl.gz \
  --out_dir lora_out_minicpm4 \
  --max_train_records 1000000 \
  --max_eval_records 20000 \
  --epochs 1 \
  --batch 8 \
  --grad_accum 4 \
  --max_seq_len 320
```

| 引数 | 既定/例 | 意味 |
| --- | --- | --- |
| `--train_path` | `../dataset/project02_2/train.jsonl.gz` | 学習データ(gzip jsonl) |
| `--eval_path` | `../dataset/project02_2/eval.jsonl.gz` | 評価データ |
| `--out_dir` | `lora_out_minicpm4` | LoRA アダプタ出力先 |
| `--max_train_records` | 1000000 | 使用する学習件数の上限 |
| `--max_eval_records` | 20000 | 使用する評価件数の上限 |
| `--epochs` | 1 | エポック数 |
| `--batch` | 8 | デバイスあたりバッチ |
| `--grad_accum` | 4 | 勾配累積(実効バッチ = 8×4=32) |
| `--max_seq_len` | 320 | プロンプト最大トークン長(超過は truncation) （プロンプト全文を含んでいるのでこれくらいの長さが必要）|

## 6. 事前準備 (データが無い場合)

```bash
# 1) 語彙 + 候補辞書
python build_vocab_from_cc100.py --max_rows 2000000

# 2) 学習/評価データ
python make_rerank_dataset_cc100.py \
  --out_train ../dataset/project02_2/train.jsonl.gz \
  --out_eval  ../dataset/project02_2/eval.jsonl.gz \
  --max_rows 2000000
```

## 7. 推論で動作確認

```bash
python infer.py \
  --lora_dir lora_out_minicpm4 \
  --ctx "今日は冬だ。" \
  --vowels "a u i"
```

> 注: 推論は generate ではなく、`prompt + 候補番号文字列` の対数尤度を候補ごとに合算して並べ替える(`score_candidate`)。多桁番号(10〜32)を正しく扱うため。


## 検証

python eval_freq_baseline.py \
  --lora_dir project02/lora_out_minicpm4 \
  --max_records 2000 \
  --vis \
  --vis_n 50