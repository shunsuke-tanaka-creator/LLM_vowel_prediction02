#!/usr/bin/env bash
# pid 1752516 (GPU 占有プロセス) の終了とデータ生成完了の両方を待ってから本番学習を自動開始する。
# 進捗は logs/auto_train.log と学習側 logs/train_YYYYMMDD_HHMMSS/train.log に出る。
set -u

PROJ_DIR="/home/shunsuke/Desktop/proj/20260609/project06"
VENV="/home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate"
WAIT_PID="${1:-1752516}"          # 終了を待つ GPU 占有プロセス
DATA_PID="${2:-}"                 # データ生成プロセス(指定時は完了も待つ)
TRAIN="${PROJ_DIR}/dataset_train" # ダミー(下で上書き)

TRAIN_PATH="../dataset/project06/train.jsonl.gz"
EVAL_PATH="../dataset/project06/eval.jsonl.gz"
LOG="${PROJ_DIR}/logs/auto_train.log"

cd "${PROJ_DIR}"
mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "${LOG}"; }

log "auto_train start. wait GPU pid=${WAIT_PID}, data pid=${DATA_PID:-none}"

# 1) データ生成プロセスの完了待ち(指定された場合)
if [ -n "${DATA_PID}" ]; then
  while kill -0 "${DATA_PID}" 2>/dev/null; do
    log "waiting dataset generation (pid=${DATA_PID})..."
    sleep 60
  done
  log "dataset generation finished."
fi

# 2) 生成物の存在確認
if [ ! -f "${TRAIN_PATH}" ] || [ ! -f "${EVAL_PATH}" ]; then
  log "ERROR: dataset files not found: ${TRAIN_PATH} / ${EVAL_PATH}. abort."
  exit 1
fi

# 3) GPU 占有プロセスの終了待ち
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  log "waiting GPU process (pid=${WAIT_PID}) to finish..."
  sleep 60
done
log "GPU process finished. checking free memory..."

# 4) 空き VRAM に応じて batch を決定(16GB 前提で最適化)
sleep 10  # メモリ解放を少し待つ
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "GPU free memory: ${FREE_MIB} MiB"

if   [ "${FREE_MIB}" -ge 14000 ]; then BATCH=16; ACCUM=2; SEQ=128
elif [ "${FREE_MIB}" -ge 11000 ]; then BATCH=8;  ACCUM=4; SEQ=128
elif [ "${FREE_MIB}" -ge 7000  ]; then BATCH=4;  ACCUM=8; SEQ=128
elif [ "${FREE_MIB}" -ge 4000  ]; then BATCH=2;  ACCUM=16; SEQ=128
else                                   BATCH=1;  ACCUM=32; SEQ=96
fi
log "chosen: batch=${BATCH} grad_accum=${ACCUM} max_seq_len=${SEQ}"

# 5) 本番学習を起動(fragmentation 緩和のため expandable_segments)
source "${VENV}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "launching training..."
python -u train_vowel_lora.py \
  --train_path "${TRAIN_PATH}" --eval_path "${EVAL_PATH}" \
  --max_train_records 2000000 --max_eval_records 20000 \
  --epochs 2 --batch "${BATCH}" --grad_accum "${ACCUM}" --max_seq_len "${SEQ}" \
  2>&1 | tee -a "${LOG}"

log "training command exited."
