#!/usr/bin/env python3
"""語尾予測の生成SFT を LoRA で学習する。プロンプト(STEM/CTX)部は loss マスクし、SUFFIX 部のみ学習する。"""
from __future__ import annotations

import argparse
import gzip
import json
from typing import Iterator, List

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


def iter_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# プロンプト本体(SUFFIX: までで生成を促す)。infer/serve と完全一致させること。
def build_prompt(stem: str, ctx: str) -> str:
    lines = []
    lines.append("あなたは日本語IMEの語尾予測器です。")
    lines.append("語幹に続く自然な語尾だけを出力してください。")
    lines.append(f"CTX: {ctx}")
    lines.append(f"STEM: {stem}")
    lines.append("SUFFIX:")
    return "\n".join(lines)


def gen_examples(path: str, max_records: int):
    n = 0
    for ex in iter_jsonl_gz(path):
        yield {"prompt": build_prompt(ex["stem"], ex.get("ctx", PROMPT_CTX_NONE)), "suffix": ex["suffix"]}
        n += 1
        if max_records > 0 and n >= max_records:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--train_path", default="../dataset/project05_02/train.jsonl.gz")
    ap.add_argument("--eval_path", default="../dataset/project05_02/eval.jsonl.gz")
    ap.add_argument("--out_dir", default="lora_suffix_out")
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

    print("Streaming train records (max=", args.max_train_records, ") from", args.train_path)
    ds_train = Dataset.from_generator(
        gen_examples,
        gen_kwargs={"path": args.train_path, "max_records": args.max_train_records},
    )
    print("Streaming eval records  (max=", args.max_eval_records, ") from", args.eval_path)
    ds_eval = Dataset.from_generator(
        gen_examples,
        gen_kwargs={"path": args.eval_path, "max_records": args.max_eval_records},
    )

    print("Loading tokenizer:", args.model_name, "rev:", args.revision)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eos = tok.eos_token_id

    # prompt 部は -100 でマスクし suffix + EOS のみ loss を取る
    def tokenize(batch):
        input_ids_list = []
        labels_list = []
        for prompt, suffix in zip(batch["prompt"], batch["suffix"]):
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            s_ids = tok(suffix, add_special_tokens=False)["input_ids"] + [eos]
            ids = (p_ids + s_ids)[: args.max_seq_len]
            labels = ([-100] * len(p_ids) + s_ids)[: args.max_seq_len]
            input_ids_list.append(ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    print("Tokenizing...")
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["prompt", "suffix"])
    ds_eval = ds_eval.map(tokenize, batched=True, remove_columns=["prompt", "suffix"])

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

    print("Loading model:", args.model_name)
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
        output_dir=args.out_dir,
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

    print("Start training...")
    trainer.train()

    print("Saving...")
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("DONE:", args.out_dir)


if __name__ == "__main__":
    main()
