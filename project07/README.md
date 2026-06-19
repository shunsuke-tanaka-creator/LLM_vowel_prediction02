# project07 — 母音列→文 復元 Web GUI

project06 の HTTP 推論 API を叩くだけの単一HTMLフロント。ビルド不要。

SSH先のサーバーで project06 の API を起動すれば、**サーバーが `GET /` でこの GUI を配信する**ので、
別PC(同一LAN / Tailscale)のブラウザからURL一つで使える。

## 構成

```
project07/
├── README.md
└── index.html   # GUI 本体 (HTML+CSS+JS 1枚)。serve_api.py の GET / が配信する
```

## 使い方(推奨: サーバー配信)

### 1. project06 の API サーバーを起動

```bash
cd ../project06
source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
LORA_DIR=logs/train_20260611_134432 uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

`serve_api.py` に追加済みのルート:
- `GET /` … この `index.html` を配信
- `POST /predict` … 推論本体

### 2. 別PCのブラウザで開く

```
http://<サーバーIP>:8000/
例) http://100.87.35.125:8000/   (Tailscale IP)
```

GUI と API が同一オリジンになり、fetch 先は相対パス `/predict`。
どのIPから開いても、開いたのと同じサーバーを自動で叩く（CORS 不要）。

- 母音列 (vowels): 空白区切り a/i/u/e/o/n
- 文脈 (ctx): 任意。空なら API 内部で `<NONE>`
- k / num_beams / max_new_tokens: パラメータ調整
- 「復元する」で TopK 候補をリスト表示

## 補足: HTMLを直接ファイルで開く場合

`index.html` を `file://` で直接開く場合、相対パス `/predict` は解決できない。
画面上の「API エンドポイント」欄を `http://<サーバーIP>:8000/predict` に書き換えること。
この場合はオリジンが異なるため CORS が必要だが、`serve_api.py` に CORS 全許可を追加済み。

