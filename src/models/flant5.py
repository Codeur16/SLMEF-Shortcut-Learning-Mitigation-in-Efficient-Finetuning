# models/flant5.py

import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# from sklearn.metrics import f1_score
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
from huggingface_hub import HfApi

# dotenv is optional at runtime; if missing, continue without failing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

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
        model_name = self.cfg.get("hf_model", self.model_name)

        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Decide whether to use 4-bit quantization based on config or experiment type
        use_4bit = bool(self.cfg.get("load_in_4bit", False)) or (self.exp_type == "efficient-finetuning")
        quant_cfg = self.cfg.get("quantization_config", {}) or {}

        try:
            if use_4bit:
                # Map dtype strings to torch dtypes
                _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                compute_dtype = _dtype_map.get(quant_cfg.get("bnb_4bit_compute_dtype", "float16"), torch.float16)

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", quant_cfg.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", False),
                )

                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=bnb_config,
                    torch_dtype=compute_dtype,
                )
            else:
                # Use dtype if specified (e.g., fp16 / bf16)
                torch_dtype = None
                cfg_dtype = str(self.cfg.get("dtype", "")).lower()
                if cfg_dtype in ("fp16", "float16"):
                    torch_dtype = torch.float16
                elif cfg_dtype in ("bf16", "bfloat16"):
                    torch_dtype = torch.bfloat16

                if torch_dtype is not None:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                    )
                else:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto",
                    )

            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        except Exception as e:
            # Fallback: try loading without quantization/dtype
            print("Warning: model loading with quantization/dtype failed, retrying without them:", e)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
            if hasattr(self.model, "gradient_checkpointing_enable"):
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

        # ----------------- Training -----------------
        print(" Starting training ...")
        start_time = time.time()
        trainer.train()
        duration = time.time() - start_time
        print(f" Training completed in {duration / 3600:.2f} hours")

        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save model and tokenizer locally
        try:
            trainer.save_model(save_dir)
        except Exception:
            # fallback to model/tokenizer save
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

        # Save training metrics: try to store last training log and evaluation
        try:
            if trainer.state.log_history:
                last_log = trainer.state.log_history[-1]
                trainer.save_metrics("train", {k: v for k, v in last_log.items() if isinstance(v, (int, float))})
        except Exception:
            pass

        try:
            metrics = trainer.evaluate()
            trainer.save_metrics("eval", metrics)
            print(" Final Evaluation:", metrics)
        except Exception as e:
            print(" Evaluation failed:", e)

        # Optionally push to Hugging Face Hub if repo & token provided
        hf_repo = os.environ.get("HF_HUB_REPO")
        hf_token = os.environ.get("HF_HUB_TOKEN")
        if hf_repo and hf_token:
            try:
                api = HfApi()
                # create repo if missing
                try:
                    api.create_repo(repo_id=hf_repo, exist_ok=True, token=hf_token)
                except Exception:
                    pass
                print(f"⬆ Pushing artifacts from {save_dir} to HuggingFace repo {hf_repo}")
                api.upload_folder(
                    folder_path=save_dir,
                    path_in_repo="",
                    repo_id=hf_repo,
                    token=hf_token,
                )
                print(" Pushed to HuggingFace Hub")
            except Exception as e:
                print(" Push to HuggingFace failed:", e)