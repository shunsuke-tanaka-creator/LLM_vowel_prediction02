#!/usr/bin/env bash
# 02(単語) / 05(語尾) / gateway(server_01) を3プロセスでまとめて起動する補助スクリプト。
# 各サーバーのログは project_server_01/ 配下に出力する。停止は Ctrl-C。
set -e

VENV=/home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
ROOT=/home/shunsuke/Desktop/proj/20260609
HERE="$ROOT/project_server_01"

source "$VENV"

# 1) 単語予測 (02)  port 8001
( cd "$ROOT/project02_2" && uvicorn serve_api:app --host 0.0.0.0 --port 8001 > "$HERE/word.log" 2>&1 ) &
WORD_PID=$!

# 2) 語尾予測 (05_02)  port 8000  本番モデル lora_suffix_out(語尾+接続/丁寧表現の拡張版)
( cd "$ROOT/project05_02" && LORA_DIR=lora_suffix_out uvicorn serve_api:app --host 0.0.0.0 --port 8000 > "$HERE/suffix.log" 2>&1 ) &
SUFFIX_PID=$!

# 3) ゲートウェイ (server_01)  port 9000
( cd "$HERE" && uvicorn gateway:app --host 0.0.0.0 --port 9000 > "$HERE/gateway.log" 2>&1 ) &
GW_PID=$!

echo "word(02) pid=$WORD_PID :8001  suffix(05) pid=$SUFFIX_PID :8000  gateway pid=$GW_PID :9000"
echo "logs: $HERE/{word,suffix,gateway}.log"

trap "kill $WORD_PID $SUFFIX_PID $GW_PID 2>/dev/null" EXIT
wait
