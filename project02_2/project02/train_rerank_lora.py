#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
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
)


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

    # コメントに追加: 大規模データ対応。take_records で全件 list 化せず、jsonl.gz をストリーミング読みしながら format_example して datasets のディスクキャッシュに直接書き出す
    print("Streaming train records (max=", args.max_train_records, ") from", args.train_path)
    ds_train = Dataset.from_generator(
        gen_formatted_texts,
        gen_kwargs={"path": args.train_path, "max_records": args.max_train_records},
    )

    print("Streaming eval records  (max=", args.max_eval_records, ") from", args.eval_path)
    ds_eval = Dataset.from_generator(
        gen_formatted_texts,
        gen_kwargs={"path": args.eval_path, "max_records": args.max_eval_records},
    )

    print("Loading tokenizer:", args.model_name, "rev:", args.revision)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True, revision=args.revision)  # コメントに追加: MiniCPM4 は custom_code モデルのため trust_remote_code=True が必須
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        out = tok(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
        )
        # コメントに追加: labels は DataCollatorForLanguageModeling(mlm=False) に作らせるため、ここでは付けない(長さ不一致で collate が失敗するため)
        return out

    print("Tokenizing...")
    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])
    ds_eval = ds_eval.map(tokenize, batched=True, remove_columns=["text"])

    print("Loading model:", args.model_name, "use_qlora:", args.use_qlora)  # コメントに追加: 0.5Bは QLoRA 不要、8B 系で有効化
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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,
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

    print("Start training...")
    trainer.train()

    print("Saving...")
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("DONE:", args.out_dir)


if __name__ == "__main__":
    main()
