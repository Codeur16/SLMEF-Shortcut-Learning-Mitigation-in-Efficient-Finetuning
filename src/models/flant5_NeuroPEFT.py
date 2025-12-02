# models/flant5.py

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)

from .base_model import BaseModel

class FlanT5Model(BaseModel):
    """
    Implementation of a fine-tunable Flan-T5 model using LoRA or QLoRA.
    This class handles:
      - model loading
      - dataset preprocessing
      - LoRA configuration
      - Seq2SeqTrainer definition
      - training + saving
    """

    def load(self):
        """Load tokenizer + model with optional quantization."""
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg["base_model"])

        if self.exp_type == "efficient-finetuning":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg["base_model"],
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg["base_model"],
                device_map="auto"
            )

        self.model.gradient_checkpointing_enable()

    def prepare_for_training(self):
        """Apply LoRA adapters or QLoRA."""
        
        if self.exp_type == "efficient-finetuning":
            self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(
            r=self.cfg["lora"]["r"],
            lora_alpha=self.cfg["lora"]["alpha"],
            lora_dropout=self.cfg["lora"]["dropout"],
            target_modules=self.cfg["lora"]["target_modules"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        self.model = get_peft_model(self.model, lora_cfg)

    def preprocess(self, examples):
        """Convert raw text into tokenized tensors."""
        MAX_INPUT = self.cfg["max_input_length"]
        MAX_TARGET = self.cfg["max_target_length"]

        inputs = [
            f"nli premise: {p} hypothesis: {h}"
            for p, h in zip(examples["premise"], examples["hypothesis"])
        ]
        labels = [self.cfg["label_map"][str(l)] for l in examples["label"]]

        model_inputs = self.tokenizer(
            inputs, max_length=MAX_INPUT, truncation=True, padding="max_length"
        )

        with self.tokenizer.as_target_tokenizer():
            model_inputs["labels"] = self.tokenizer(
                labels, max_length=MAX_TARGET, truncation=True, padding="max_length"
            ).input_ids

        return model_inputs

    def train(self, train_dataset, val_dataset, save_dir):
        """Perform full fine-tuning."""

        tokenized_train = train_dataset.map(self.preprocess, batched=True)
        tokenized_val = val_dataset.map(self.preprocess, batched=True)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=self.cfg["learning_rate"],
            num_train_epochs=self.cfg["epochs"],
            per_device_train_batch_size=self.cfg["batch_size"],
            per_device_eval_batch_size=self.cfg["batch_size"],
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            predict_with_generate=True,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
