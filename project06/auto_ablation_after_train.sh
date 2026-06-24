#!/bin/bash
# 学習完了(DONEログ + adapterファイル)を検知したら自動でアブレーションを起動するランナー。
LOG="/home/shunsuke/Desktop/proj/20260609/project06/logs/train_v03_run.log"
PROJ="/home/shunsuke/Desktop/proj/20260609/project06"
RUN_DIR="$PROJ/logs/train_20260622_140822"
VENV="/home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate"
ABL_LOG="$PROJ/logs/ablation_v03_run.log"

cd "$PROJ"
# 学習完了を待つ: DONE ログが出て、かつ adapter ファイルが保存されるまで
while true; do
  if grep -q "DONE:" "$LOG" 2>/dev/null && ls "$RUN_DIR"/adapter_model.* >/dev/null 2>&1; then
    break
  fi
  sleep 60
done

echo "学習完了を検知。アブレーションを開始します: $(date)" | tee "$ABL_LOG"
source "$VENV"
bash run_ablation.sh 2>&1 | tee -a "$ABL_LOG"
echo "ABLATION_DONE: $(date)" | tee -a "$ABL_LOG"
