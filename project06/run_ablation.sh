#!/usr/bin/env bash
# アブレーション一括実行。a~e を直列で dump_predictions -> evaluate_for_paper まで回し、
# 最後に summary.csv に横並びでまとめる。全ログは run.log に tee する。
#   source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
#   bash run_ablation.sh
set -euo pipefail

LORA_DIR="logs/train_20260622_140822"
EVAL_PATH="../dataset/project06/eval03.jsonl.gz"
MAX_RECORDS=0   # 0 で全件
OUT_DIR="paper/results/ablation_20260623"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

# 1ラン分: dump_predictions -> evaluate_for_paper
# 引数: tag  ctx_flag(--use_ctx か空)  beam(=k)  use_lora(1=LoRAあり / 0=素のベース)
run_one() {
  local tag="$1" ctx_flag="$2" beam="$3" use_lora="$4"
  echo "===== [$tag] ctx=${ctx_flag:-none} beam=$beam lora=$use_lora ====="
  local model_args
  if [ "$use_lora" = "1" ]; then
    model_args="--lora_dir $LORA_DIR"
  else
    model_args="--no_lora"
  fi
  python -u paper/scripts/dump_predictions.py \
    $model_args \
    --eval_path "$EVAL_PATH" \
    --max_records "$MAX_RECORDS" \
    --k "$beam" --num_beams "$beam" \
    $ctx_flag \
    --out "$OUT_DIR/preds_${tag}.jsonl"
  python -u paper/scripts/evaluate_for_paper.py \
    --pred_path "$OUT_DIR/preds_${tag}.jsonl" \
    --out_dir "$OUT_DIR" --tag "$tag"
}

{
  #        tag  ctx          beam  use_lora
  run_one  a    "--use_ctx"  8     1
  run_one  b    ""           8     1
  run_one  c    "--use_ctx"  4     1
  run_one  d    "--use_ctx"  16    1
  run_one  e    "--use_ctx"  8     0

  # summary.csv: 5本の metrics_*.json を横並びにまとめる
  python - "$OUT_DIR" <<'PY'
import json, os, sys, glob, csv
out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "metrics_*.json"))):
    tag = os.path.basename(path)[len("metrics_"):-len(".json")]
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m = {"tag": tag, **m}
    rows.append(m)
if rows:
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("summary ->", os.path.join(out_dir, "summary.csv"))
PY
} 2>&1 | tee "$LOG"
