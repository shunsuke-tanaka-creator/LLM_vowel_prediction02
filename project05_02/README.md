# project05_02 — 語尾・接続/丁寧表現予測 AI モデル

語幹(例 `寒` / `目立` / `学生`)から、続く活用語尾・接続表現・丁寧表現を複数生成するモデル。
日本語IMEで語尾を選ぶときに出てくる候補のような体験を、`MiniCPM4-0.5B` + LoRA で実現する。
project05 の「活用語尾のみ」から、接続助詞(ので/から/けど)・断定/丁寧(です/だ/ます)・終助詞(だよね)まで拡張した版。

## できること

- 入力: 語幹(stem) + 任意の文脈(ctx)
- 出力: 続く語尾・接続/丁寧表現候補 TopK(自由生成)

| 入力 stem | 出力(候補の例) |
| --- | --- |
| 寒 | い / かった / そう / くない / かったので |
| 目立 | った / っている |
| 走 | った / っていた / ります / りますけど |
| 行く | から / けど |
| 学生 | です / だよね |
| 元気 | だよね / だから |

`stem` のみ / `stem + ctx` の両方に対応(ctx 無しは内部で `<NONE>`)。

## 設計

- 形式: 自由生成(SFT)。プロンプト `... STEM: 寒\nSUFFIX:` → モデルが `かった` を生成。
- 語幹定義:
  - 用言(動詞/形容詞): 辞書形ベース。表層 `寒かっ`(辞書形 `寒い`)の共通接頭辞 `寒` を stem、残り `かっ` + 後続助動詞 `た` を suffix とする。
  - 名詞/形状詞(形容動詞): 表層そのものを stem とし、後続の断定助動詞・助詞を suffix とする(例 `学生` → `です`、`元気` → `だよね`)。後続が無い単独名詞はノイズなので除外。
- 抽出は UniDic(fugashi)で行い、用言・名詞・形状詞を起点に、後続の助動詞・接続助詞(て/ば/けど/から/ので/のに/が/し)・終助詞(ね/よ/よね)・準体助詞(の)・補助動詞(いる/ある等)・形状詞の助動詞語幹(そう)をまとめて suffix とする。
- 接続助詞は表層リストではなく UniDic の `pos2`(接続助詞/終助詞/準体助詞)で判定し、格助詞「で」「に」(全力で/家に)を巻き込まないようにしている。
- 名詞起点では後続の自立用言はマージしない(「昨日」+「来た」のような別文節のノイズを防ぐ)。

```mermaid
graph LR
  raw["原文: 学生だよね / 寒いので"] --> morph["UniDic 形態素解析"]
  morph --> stem["stem: 学生 / 寒"]
  morph --> suffix["suffix: だよね / いので"]
  stem --> sft["SFT(LoRA): prompt→suffix"]
  suffix --> sft
  sft --> gen["生成: 学生→です/だよね, 寒→い/ので/かった"]
```

## ファイル構成

```
project05_02/
├── README.md
├── requirements.txt
├── tanakamemo.txt                 # train/infer コマンドメモ
├── morph_suffix_utils.py          # stem/suffix 抽出 (fugashi + unidic)
├── make_suffix_dataset_cc100.py   # CC100 → ../dataset/project05_02/{train,eval}.jsonl.gz を生成
├── train_suffix_lora.py           # 語尾SFT 学習 (suffix部のみ loss マスク)
├── infer.py                       # CLI 推論
└── serve_api.py                   # HTTP 推論 (POST /predict)
```

## セットアップ

```bash
source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
pip install -r requirements.txt
python -m unidic download   # 形態素解析辞書(初回のみ)
```

## 使い方

データ生成 → 学習 → 推論 の順。詳細コマンドは `tanakamemo.txt` を参照。

### データ生成

```bash
python make_suffix_dataset_cc100.py --max_rows 2000000
```

学習/評価データは `../dataset/project05_02/{train,eval}.jsonl.gz` に出力される。

### 学習

```bash
python train_suffix_lora.py \
  --train_path ../dataset/project05_02/train.jsonl.gz --eval_path ../dataset/project05_02/eval.jsonl.gz \
  --out_dir lora_suffix_out \
  --max_train_records 2000000 --epochs 2 --batch 8 --grad_accum 4
```

### 推論 (CLI)

```bash
python infer.py --lora_dir lora_suffix_out --stem "寒" --ctx "今日は" --k 8
python infer.py --lora_dir lora_suffix_out --stem "学生" --ctx "私は" --k 8
```

### 推論 (HTTP API)

```bash
LORA_DIR=lora_suffix_out uvicorn serve_api:app --host 0.0.0.0 --port 8000
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"stem":"寒","ctx":"今日は","k":8}'
```

レスポンス例:

```json
{"stem":"寒","candidates":["かった","い","そう"],"words":["寒かった","寒い","寒そう"]}
```

## データ形式 (jsonl.gz)

```json
{"stem": "寒", "ctx": "今日は", "suffix": "かった"}
{"stem": "目立", "ctx": "その建物は", "suffix": "っている"}
{"stem": "学生", "ctx": "私は", "suffix": "です"}
{"stem": "元気", "ctx": "彼は", "suffix": "だよね"}
```

## 主なハイパーパラメータ

| 項目 | デフォルト | 説明 |
| --- | --- | --- |
| ctx_len | 4 | CTX に含める直前の用言数 |
| max_suffix_len | 16 | 語尾の最大文字数(接続助詞/終助詞付きで長くなるため 12→16)|
| none_ctx_ratio | 0.3 | CTX を `<NONE>` にする割合(語幹のみ入力対応) |
| max_seq_len | 128 | 学習時の最大トークン長 |
| num_beams / k | 8 / 8 | 推論のビーム幅・候補数 |
