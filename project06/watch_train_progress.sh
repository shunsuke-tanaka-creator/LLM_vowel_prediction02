#!/bin/bash
# 学習ログから進捗だけを抜き出して PROGRESS.txt に1分ごとに書き出す監視スクリプト
LOG="${1:-/home/shunsuke/.cursor/projects/home-shunsuke-Desktop-proj-20260609-project06/terminals/215149.txt}"
OUT="${2:-/home/shunsuke/Desktop/proj/20260609/project06/PROGRESS.txt}"

while true; do
  # 最新の進捗バー(step/total [経過<残り, it/s])を取得。total が最大=学習本体のバーだけを採用(eval の 188 バーを除外)
  last=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+(it/s|s/it)\]" "$LOG" 2>/dev/null \
         | awk -F'[/ ]' '{print $2, $0}' | sort -n | awk '{$1=""; print substr($0,2)}' \
         | grep -E "/46876|/[0-9]{5,}" | tail -1)
  [ -z "$last" ] && last=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+(it/s|s/it)\]" "$LOG" 2>/dev/null | tail -1)
  # 最新の loss 行
  loss=$(grep -oE "\{'loss':[^}]*\}" "$LOG" 2>/dev/null | tail -1)
  eval=$(grep -oE "\{'eval_loss':[^}]*\}" "$LOG" 2>/dev/null | tail -1)
  done=$(grep -E "DONE:|Saved|saved adapter" "$LOG" 2>/dev/null | tail -1)
  {
    echo "更新時刻: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "進捗     : $last"
    echo "loss     : $loss"
    echo "eval     : $eval"
    [ -n "$done" ] && echo "完了     : $done"
  } > "$OUT"
  [ -n "$done" ] && break
  sleep 60
done
