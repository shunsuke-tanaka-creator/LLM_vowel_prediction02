# paper — 母音入力 + LoRA LLM 日本語入力 論文一式

`project06`（母音列→文の生成SFT）の実装に基づく学会論文ドラフトと、論文用の評価・作図コード。

## ディレクトリ

```
paper/
  main.tex                 # 本体（jarticle, twocolumn, 4ページ想定）
  refs.bib                 # 参考文献（[要確認] は文献調査で確定）
  sections/                # 各章 .tex
  figures/                 # 図（PNG）
  tables/                  # 表データ（CSV; 本文の表は .tex 内に直接記述）
  results/                 # 評価結果（metrics_*.json, errors_*.csv, preds_*.jsonl）
  scripts/
    dump_predictions.py    # project06 モデルで推論し評価用 JSONL を出力
    evaluate_for_paper.py  # Acc@k/MRR/EM/CER/WER/KSPC とエラーCSVを計算
    plot_paper_figures.py  # 比較グラフ等を生成（実値が無ければ仮値で動作）
    make_concept_figures.py# 概要図・処理フロー図を生成
```

## 論文のビルド方法

```bash
cd paper
platex main.tex
pbibtex main
platex main.tex
platex main.tex
dvipdfmx main.dvi   # -> main.pdf
```
（uplatex/lualatex-ja 等を使う場合は documentclass を適宜変更）

## 評価コードの実行方法

venv を有効化してから実行する（ローカル環境に直接入れない）。

```bash
source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate

# 1) project06 ルートで推論結果を JSONL に出力（GPU 必要）
cd /home/shunsuke/Desktop/proj/20260609/project06
python paper/scripts/dump_predictions.py \
  --lora_dir logs/train_20260611_134432 \
  --eval_path ../dataset/project06/eval.jsonl.gz \
  --max_records 1309 --k 8 \
  --out paper/results/preds_proposed.jsonl

# 2) 指標を計算（CPUのみで可）
python paper/scripts/evaluate_for_paper.py \
  --pred_path paper/results/preds_proposed.jsonl \
  --out_dir paper/results --tag proposed
```

`evaluate_for_paper.py` の入力 JSONL は
`{"input_vowels","gold","candidates","scores"}` 形式
（`vowels/sentence/cands` などの別名も自動吸収）。

## グラフ生成方法

```bash
cd paper/scripts
python make_concept_figures.py     # fig_overview.png, fig_flow.png
python plot_paper_figures.py       # 比較グラフ一式（results があれば実値）
```

## `XXX` を差し替える場所

- 本文中の `XXX`：Acc@1/3/5, MRR, CER, WER, KSPC, 入力削減率, 入力時間, NASA-TLX, SUS, 画面占有率
- 表 `tab:acc`, `tab:abl`, `tab:input` の `XXX`
- `plot_paper_figures.py` の `SAMPLE_*`（`TODO: replace with actual scores`）
- 数値は `results/metrics_*.json` を算出後に転記する。

## 実ユーザ実験が必要な項目

入力時間 / 1文字あたり入力時間 / 修正回数 / 候補選択回数 / 誤入力率 /
タスク完了率 / NASA-TLX / SUS / 若年・高齢比較 / フリック熟練・非熟練比較 /
画面占有率の実測 / KSPC の実測。詳細は `TODO.md`。

## 引用確認が必要な箇所

`refs.bib` の `note = {[要確認]}` を持つ全エントリ
（フリック入力、スマートウォッチ入力、曖昧入力、Dasher、視線入力、VR入力、日本語IME）。
LoRA / NASA-TLX / SUS / KSPC / Soukoreff は書誌を記載済みだが最終確認すること。
