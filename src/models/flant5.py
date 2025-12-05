"""
FlanT5 Model Implementation for Efficient Fine-tuning

This module provides a FlanT5Model class that supports:
- QLoRA (4-bit quantization) for efficient fine-tuning
- LoRA adapters for parameter-efficient training
- Sequence-to-sequence training with proper label tokenization
- Model saving and optional HuggingFace Hub integration
"""

from datetime import datetime
import os
import time
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
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
        
        # If checkpoint is provided, load from checkpoint
        if self.checkpoint and os.path.exists(self.checkpoint):
            print(f"Loading model from checkpoint: {self.checkpoint}")
            # Load tokenizer from checkpoint
            self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint)
        else:
            # Load tokenizer from base model
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Decide whether to use 4-bit quantization based on config or experiment type
        use_4bit = bool(self.cfg.get("load_in_4bit", False)) or (self.exp_type == "efficient-finetuning")
        quant_cfg = self.cfg.get("quantization_config", {}) or {}

        # Decide whether to use 4-bit quantization based on config or experiment type
        use_4bit = bool(self.cfg.get("load_in_4bit", False)) or (self.exp_type == "efficient-finetuning")
        quant_cfg = self.cfg.get("quantization_config", {}) or {}
        
        # If checkpoint is provided, load model from checkpoint
        if self.checkpoint and os.path.exists(self.checkpoint):
            try:
                # Try loading as PEFT model first (for LoRA adapters)
                from peft import PeftModel
                # Load base model first
                base_model_name = model_name
                
                if use_4bit:
                    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                    compute_dtype = _dtype_map.get(quant_cfg.get("bnb_4bit_compute_dtype", "float16"), torch.float16)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", False),
                    )
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model_name,
                        device_map="auto",
                        quantization_config=bnb_config,
                        dtype=compute_dtype,
                    )
                else:
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model_name,
                        device_map="auto",
                    )
                
                # Try to load PEFT adapters
                try:
                    self.model = PeftModel.from_pretrained(base_model, self.checkpoint)
                    print(" Loaded PEFT model with adapters from checkpoint")
                except Exception:
                    # If PEFT loading fails, try loading as regular model
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.checkpoint,
                        device_map="auto",
                    )
                    print(" Loaded model from checkpoint (non-PEFT)")
            except Exception as e:
                print(f"Warning: Failed to load from checkpoint, loading base model: {e}")
                # Fallback to base model
                self.checkpoint = None
        
        # Load base model if no checkpoint or checkpoint loading failed
        if not (self.checkpoint and os.path.exists(self.checkpoint)):
            try:
                if use_4bit:
                    # Map dtype strings to torch dtypes
                    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                    compute_dtype = _dtype_map.get(quant_cfg.get("bnb_4bit_compute_dtype", "float16"), torch.float16)

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", False),
                    )

                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        quantization_config=bnb_config,
                        dtype=compute_dtype,
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
                            dtype=torch_dtype,
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
        if self.model is None:
            raise ValueError("Model must be loaded before preparing for training. Call load() first.")
        
        if self.exp_type == "efficient-finetuning":
            self.model = prepare_model_for_kbit_training(self.model)

        qlora_config = self.cfg.get("qlora", {})
        qlora_cfg = LoraConfig(
            r=qlora_config.get("r", 16),
            lora_alpha=qlora_config.get("alpha", 32),
            lora_dropout=qlora_config.get("dropout", 0.05),
            target_modules=qlora_config.get("target_modules", ["q", "k", "v", "o"]),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        self.model = get_peft_model(self.model, qlora_cfg)

    def preprocess(self, examples):
        """Convert raw text into tokenized tensors."""
        MAX_INPUT = self.cfg["max_input_length"]
        MAX_TARGET = self.cfg["max_target_length"]

        inputs = [
            f"nli premise: {p} hypothesis: {h}"
            for p, h in zip(examples["premise"], examples["hypothesis"])
        ]
        
        # Map labels to text labels
        label_map = self.cfg.get("label_map", {})
        labels = [label_map.get(str(l), "unknown") for l in examples["label"]]

        model_inputs = self.tokenizer(
            inputs, max_length=MAX_INPUT, truncation=True, padding="max_length"
        )

        # Use text_target parameter instead of deprecated as_target_tokenizer()
        labels_tokenized = self.tokenizer(
            text_target=labels,
            max_length=MAX_TARGET,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels_tokenized["input_ids"]

        return model_inputs

    def train(self, train_dataset, val_dataset, save_dir):
        """Perform full fine-tuning."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before training. Call load() first.")

        tokenized_train = train_dataset.map(
            self.preprocess, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_val = val_dataset.map(
            self.preprocess, 
            batched=True,
            remove_columns=val_dataset.column_names
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        # Check if CUDA is available for optimization
        use_fp16 = torch.cuda.is_available()
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=float(self.cfg["learning_rate"]),
            num_train_epochs=int(self.cfg["epochs"]),
            per_device_train_batch_size=int(self.cfg["batch_size"]),
            per_device_eval_batch_size=int(self.cfg["batch_size"]),
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            predict_with_generate=True,
            report_to="none",
            fp16=use_fp16 and not use_bf16,  # Use FP16 if CUDA available but not BF16
            bf16=use_bf16,  # Use BF16 if supported (better than FP16)
            dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory only if GPU available
            gradient_accumulation_steps=1,  # Can increase if memory allows
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )

        # ----------------- Training -----------------
        print("Starting training ...")
        start_time = time.time()
        try:
            trainer.train()
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        duration = time.time() - start_time
        print(f"Training completed in {duration / 3600:.2f} hours")

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

        # Optionally push to Hugging Face Hub if configured
        push_to_hub = bool(self.cfg.get("push_to_hub", False))
        hf_repo = os.environ.get("HF_HUB_REPO") or self.cfg.get("hf_repo")
        hf_token = os.environ.get("HF_HUB_TOKEN") or self.cfg.get("hf_token")

        # -------------------------------------------------------------------
        # Optional: Create a timestamped backup copy in saved_models (if desired)
        # Note: The main save_dir (parameter) is already used above for saving
        create_backup = self.cfg.get("create_backup_copy", False)
        if create_backup:
            model_name = self.cfg.get("model_name", "model")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join("saved_models", model_name, timestamp)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy model files to backup directory
            import shutil
            try:
                for file in os.listdir(save_dir):
                    src = os.path.join(save_dir, file)
                    dst = os.path.join(backup_dir, file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                print(f" Backup copy created in {backup_dir}")
            except Exception as e:
                print(f" Warning: Could not create backup copy: {e}")

        # -------------------------------------------------------------------
        # Push to HuggingFace Hub (use the original save_dir, not backup)
        if push_to_hub:
            if not hf_repo or not hf_token or hf_token.strip() == "" or hf_token == "your_huggingface_token_here ":
                print("⚠️ Push to HuggingFace is enabled but HF_HUB_REPO or HF_HUB_TOKEN are missing/invalid")
                print("   Set HF_HUB_REPO and HF_HUB_TOKEN environment variables or configure in YAML")
            else:
                try:
                    api = HfApi()
                    print(f"📤 Creating repository {hf_repo} if it doesn't exist...")
                    api.create_repo(repo_id=hf_repo, exist_ok=True, token=hf_token)

                    print(f"📤 Uploading folder {save_dir} to HuggingFace Hub...")
                    api.upload_folder(
                        folder_path=save_dir,
                        repo_id=hf_repo,
                        token=hf_token,
                        path_in_repo="",  # root
                    )

                    print(" Upload completed successfully on HuggingFace Hub")

                except Exception as e:
                    print(f"❌ Error when sending to HuggingFace: {e}")
        else:
            print("ℹ️ Push to HuggingFace disabled (`push_to_hub=False`).")
    
    def inference(self, dataset, output_file):
        """Run inference over a dataset and save predictions."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before inference. Call load() first.")
        
        import json
        from tqdm import tqdm
        
        # Prepare inputs
        inputs = [
            f"nli premise: {p} hypothesis: {h}"
            for p, h in zip(dataset["premise"], dataset["hypothesis"])
        ]
        
        # Run inference in batches
        batch_size = 32
        predictions = []
        
        print(f"Running inference on {len(inputs)} examples...")
        for i in tqdm(range(0, len(inputs), batch_size)):
            batch_inputs = inputs[i:i + batch_size]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                batch_inputs,
                max_length=self.cfg["max_input_length"],
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_length=self.cfg["max_target_length"],
                    num_beams=1,
                    do_sample=False
                )
            
            # Decode predictions
            decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)
        
        # Prepare output data
        output_data = []
        label_map = self.cfg.get("label_map", {})
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        for i, (premise, hypothesis, label, pred) in enumerate(zip(
            dataset["premise"],
            dataset["hypothesis"],
            dataset["label"],
            predictions
        )):
            # Map prediction text to label id
            pred_label_id = reverse_label_map.get(pred.lower().strip(), -1)
            true_label_id = int(label) if isinstance(label, (int, str)) else label
            
            output_data.append({
                "id": i,
                "premise": premise,
                "hypothesis": hypothesis,
                "true_label": int(true_label_id),
                "true_label_text": label_map.get(str(true_label_id), "unknown"),
                "predicted_label": int(pred_label_id) if pred_label_id != -1 else None,
                "predicted_label_text": pred.strip(),
            })
        
        # Save to JSON file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f" Inference completed. Predictions saved to {output_file}")
        return output_data