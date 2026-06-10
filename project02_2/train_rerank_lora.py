#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import logging  # 追加: 学習ログをファイルに残すため
import os  # 追加: ログ保存ディレクトリ作成のため
from datetime import datetime  # 追加: ログファイル名に日付時刻を付与するため
from typing import Iterator, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,  # 追加: 自前 labels を尊重してパディングする collator
)


# 追加: 標準出力 + 日付時刻付きログファイルの両方に出力するロガーを用意する
def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, datetime.now().strftime("train_%Y%m%d_%H%M%S.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


# 追加: trainer.state.log_history から train/eval loss を取り出して学習曲線を画像保存する
def save_loss_curve(log_history: list, out_png: str):
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for rec in log_history:
        if "loss" in rec and "step" in rec:
            train_steps.append(rec["step"])
            train_loss.append(rec["loss"])
        if "eval_loss" in rec and "step" in rec:
            eval_steps.append(rec["step"])
            eval_loss.append(rec["eval_loss"])

    import matplotlib
    matplotlib.use("Agg")  # 追加: GUI 無し環境でも保存できるようにする
    import matplotlib.pyplot as plt

    plt.figure()
    if train_loss:
        plt.plot(train_steps, train_loss, label="train_loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("training curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def iter_jsonl_gz(path: str) -> Iterator[dict]:
    """jsonl.gz を逐次yield（全件をメモリに載せない）"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def take_records(path: str, max_records: int) -> List[dict]:
    """最大 max_records 件だけ読み込む（0なら全件…だが巨大なので非推奨）"""
    out: List[dict] = []
    if max_records <= 0:
        # 互換性のため残すが、巨大データだとメモリ死にます
        for ex in iter_jsonl_gz(path):
            out.append(ex)
        return out

    for i, ex in enumerate(iter_jsonl_gz(path), start=1):
        out.append(ex)
        if i >= max_records:
            break
    return out


def format_example(ex: dict) -> str:
    """
    multi-choice：出力は番号のみ
    ex = {"ctx":..., "vowels":..., "cands":[...], "answer":"7"}
    """
    lines = []
    lines.append("あなたは日本語IMEの予測器です。")
    lines.append("出力は候補番号のみ。")
    lines.append(f"CTX: {ex['ctx']}")
    lines.append(f"VOWELS: {ex['vowels']}")
    lines.append("CANDIDATES:")
    for i, w in enumerate(ex["cands"], start=1):
        lines.append(f"{i}) {w}")
    lines.append("ANSWER:")
    lines.append(str(ex["answer"]))
    return "\n".join(lines)


# コメントに追加: 大規模データ対応。全件 list 化せずに jsonl.gz を逐次 yield してそのまま format_example したテキストを返すジェネレータ。Dataset.from_generator と組み合わせて使う。
def gen_formatted_texts(path: str, max_records: int):
    """jsonl.gz を逐次読みつつ format_example したテキストだけを yield する。"""
    n = 0
    for ex in iter_jsonl_gz(path):
        yield {"text": format_example(ex)}
        n += 1
        if max_records > 0 and n >= max_records:
            break


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--train_path", default="../../dataset/project02_2/train.jsonl.gz")
    ap.add_argument("--eval_path", default="../../dataset/project02_2/eval.jsonl.gz")
    ap.add_argument("--out_dir", default="lora_rerank_out")
    ap.add_argument("--log_dir", default="logs")  # 追加: 学習ログ(日付時刻付き)の保存先
    # コメントに追加: QLoRA(4bit)を使うかどうか。0.5Bでは不要、8B系で有効化推奨
    ap.add_argument("--use_qlora", action="store_true", default=False)
    # コメントに追加: HFモデルのリビジョン pin。MiniCPM4-0.5B 最新(5253c7f)は transformers>=4.50 が必須なため、4.46.3 環境では古い rev を使う
    ap.add_argument("--revision", default="2aaa97c53d")

    # ★ここが今回の肝（メモリ対策）
    ap.add_argument("--max_train_records", type=int, default=5_000_000)  # コメントに追加: 100万→500万に増量
    ap.add_argument("--max_eval_records", type=int, default=20_000)

    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)  # コメントに追加: 0.5→2.0 に増量
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()

    # 追加: 日付時刻付きログファイルへの保存を開始
    log_path = setup_logging(args.log_dir)
    import transformers  # 追加: 学習中の loss/eval ログもファイルに残すため
    transformers.utils.logging.set_verbosity_info()  # 追加: Trainer のログを INFO で root ロガー経由でファイル出力
    logging.info("Log file: %s", log_path)
    logging.info("Args: %s", vars(args))

    # コメントに追加: 大規模データ対応。take_records で全件 list 化せず、jsonl.gz をストリーミング読みしながら format_example して datasets のディスクキャッシュに直接書き出す
    logging.info("Streaming train records (max=%s) from %s", args.max_train_records, args.train_path)
    ds_train = Dataset.from_generator(
        gen_formatted_texts,
        gen_kwargs={"path": args.train_path, "max_records": args.max_train_records},
    )

    logging.info("Streaming eval records (max=%s) from %s", args.max_eval_records, args.eval_path)
    ds_eval = Dataset.from_generator(
        gen_formatted_texts,
        gen_kwargs={"path": args.eval_path, "max_records": args.max_eval_records},
    )

    logging.info("Loading tokenizer: %s rev: %s", args.model_name, args.revision)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True, revision=args.revision)  # コメントに追加: MiniCPM4 は custom_code モデルのため trust_remote_code=True が必須
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        # 追加: design通り「ANSWER:\n より後ろ(答え番号+EOS)だけに loss をかける」ため labels を自前作成する
        input_ids_list, labels_list = [], []  # 追加
        for text in batch["text"]:  # 追加
            prompt, answer = text.rsplit("ANSWER:\n", 1)  # 追加: ANSWER:\n を境界に分割
            prompt = prompt + "ANSWER:\n"  # 追加
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"]  # 追加
            a_ids = tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]  # 追加: 末尾に EOS
            ids = (p_ids + a_ids)[: args.max_seq_len]  # 追加
            labels = ([-100] * len(p_ids) + a_ids)[: args.max_seq_len]  # 追加: prompt をマスク
            input_ids_list.append(ids)  # 追加
            labels_list.append(labels)  # 追加
        return {"input_ids": input_ids_list, "labels": labels_list}  # 追加

    logging.info("Tokenizing...")
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_eval = ds_eval.map(tokenize, batched=True, remove_columns=["text"])

    logging.info("Loading model: %s use_qlora: %s", args.model_name, args.use_qlora)  # コメントに追加: 0.5Bは QLoRA 不要、8B 系で有効化
    if args.use_qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  # コメントに追加: MiniCPM4 用
            revision=args.revision,  # コメントに追加: rev pin
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  # コメントに追加: MiniCPM4 用
            revision=args.revision,  # コメントに追加: rev pin
        )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # LoRA: attention + MLP (all-linear)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(  # 変更: 自前 labels を尊重してパディングする
        tokenizer=tok,
        padding=True,
        label_pad_token_id=-100,  # 追加: パディングした labels も loss から除外
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=2000,  # コメントに追加: 500→2000 (長時間学習向けに評価間隔を広げる)
        save_steps=2000,  # コメントに追加: 500→2000 (チェックポイント書き出しを減らす)
        save_total_limit=3,  # コメントに追加: 2→3 (長時間学習で世代を1つ多めに残す)
        bf16=True,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",  # コメントに追加: QLoRA 無効時は通常 AdamW
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
    )

    logging.info("Start training...")
    trainer.train()

    # 追加: 学習曲線(train/eval loss)を画像として保存。ログと同じタイムスタンプ名にする
    curve_png = os.path.splitext(log_path)[0] + "_curve.png"
    save_loss_curve(trainer.state.log_history, curve_png)
    logging.info("Loss curve saved: %s", curve_png)

    logging.info("Saving...")
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    logging.info("DONE: %s", args.out_dir)


if __name__ == "__main__":
    main()
