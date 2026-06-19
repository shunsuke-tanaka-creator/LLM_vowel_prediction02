#!/usr/bin/env python3
"""母音列→文 復元の生成SFT を LoRA で学習する。プロンプト(CTX/VOWELS)部は loss マスクし、SENTENCE 部のみ学習する。"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from datetime import datetime
from typing import Iterator

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

PROMPT_CTX_NONE = "<NONE>"


# 標準出力 + 日付時刻付きフォルダ(train_YYYYMMDD_HHMMSS)にログ・曲線・重みを保存。run_dir を返す
def setup_logging(log_dir: str) -> str:
    run_dir = os.path.join(log_dir, datetime.now().strftime("train_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return run_dir


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
    matplotlib.use("Agg")
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
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# プロンプト本体(SENTENCE: までで生成を促す)。infer/serve と完全一致させること。
def build_prompt(vowels: str, ctx: str) -> str:
    lines = []
    lines.append("あなたは日本語IMEの文復元器です。")
    lines.append("母音列(a/i/u/e/o/n)から元の文を復元してください。")
    lines.append(f"CTX: {ctx}")
    lines.append(f"VOWELS: {vowels}")
    lines.append("SENTENCE:")
    return "\n".join(lines)


def gen_examples(path: str, max_records: int):
    n = 0
    for ex in iter_jsonl_gz(path):
        yield {"prompt": build_prompt(ex["vowels"], ex.get("ctx", PROMPT_CTX_NONE)), "sentence": ex["sentence"]}
        n += 1
        if max_records > 0 and n >= max_records:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--train_path", default="../dataset/project06/train.jsonl.gz")
    ap.add_argument("--eval_path", default="../dataset/project06/eval.jsonl.gz")
    ap.add_argument("--out_dir", default="lora_vowel_out")  # 保存先は logs/train_YYYYMMDD_HHMMSS/ に一本化(後方互換で残す)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--revision", default="2aaa97c53d")  # transformers 4.46.3 互換 rev pin

    ap.add_argument("--max_train_records", type=int, default=2_000_000)
    ap.add_argument("--max_eval_records", type=int, default=20_000)

    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = setup_logging(args.log_dir)
    import transformers
    transformers.utils.logging.set_verbosity_info()
    logging.info("Run dir: %s", run_dir)
    logging.info("Args: %s", vars(args))

    logging.info("Streaming train records (max=%s) from %s", args.max_train_records, args.train_path)
    ds_train = Dataset.from_generator(
        gen_examples,
        gen_kwargs={"path": args.train_path, "max_records": args.max_train_records},
    )
    logging.info("Streaming eval records (max=%s) from %s", args.max_eval_records, args.eval_path)
    ds_eval = Dataset.from_generator(
        gen_examples,
        gen_kwargs={"path": args.eval_path, "max_records": args.max_eval_records},
    )

    logging.info("Loading tokenizer: %s rev: %s", args.model_name, args.revision)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eos = tok.eos_token_id

    # prompt 部は -100 でマスクし sentence + EOS のみ loss を取る
    def tokenize(batch):
        input_ids_list = []
        labels_list = []
        for prompt, sentence in zip(batch["prompt"], batch["sentence"]):
            p_ids = tok(prompt + "\n", add_special_tokens=True)["input_ids"]  # SENTENCE:\n までを prompt
            s_ids = tok(sentence, add_special_tokens=False)["input_ids"] + [eos]
            ids = (p_ids + s_ids)[: args.max_seq_len]
            labels = ([-100] * len(p_ids) + s_ids)[: args.max_seq_len]
            input_ids_list.append(ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    logging.info("Tokenizing...")
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["prompt", "sentence"])
    ds_eval = ds_eval.map(tokenize, batched=True, remove_columns=["prompt", "sentence"])

    pad_id = tok.pad_token_id

    # input_ids/labels を右パディングする collator(labels の pad は -100)
    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attn = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    logging.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=args.revision,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

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

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate,
    )

    logging.info("Start training...")
    trainer.train()

    curve_png = os.path.join(run_dir, "curve.png")
    save_loss_curve(trainer.state.log_history, curve_png)
    logging.info("Loss curve saved: %s", curve_png)

    logging.info("Saving...")
    trainer.save_model(run_dir)
    tok.save_pretrained(run_dir)
    logging.info("DONE: %s", run_dir)


if __name__ == "__main__":
    main()
