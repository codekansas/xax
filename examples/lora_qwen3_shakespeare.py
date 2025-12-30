"""LoRA fine-tuning of Qwen3-4B on Shakespeare using the xax LoRA helpers."""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from xax.nn.lora import torch_loraize, torch_merge_lora


def prepare_dataset(tokenizer: AutoTokenizer, max_length: int = 512) -> torch.utils.data.Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = load_dataset("tiny_shakespeare", split="train")
    return ds.map(_tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    model_id = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_ds = prepare_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    torch_loraize(
        model,
        rank=8,
        alpha=32.0,
        dropout_rate=0.05,
        predicate=lambda lin: lin.out_features >= 1024,
    )

    training_args = TrainingArguments(
        output_dir="qwen3-4b-shakespeare-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        learning_rate=2e-4,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )
    trainer.train()

    torch_merge_lora(model)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
