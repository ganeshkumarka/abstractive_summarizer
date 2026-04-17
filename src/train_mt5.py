# """
# src/train_mt5.py
# ----------------
# Fine-tune google/mt5-small for Malayalam abstractive summarization.

# mt5-small is a multilingual T5 model (300M params) pretrained on Common Crawl
# in 101 languages including Malayalam. Fine-tuning it for summarization on
# Social-Sum-Mal gives dramatically better results than training from scratch.

# Why this works where frozen MuRIL didn't:
#   - MuRIL = encoder only (BERT-style), not designed for generation
#   - mt5 = encoder-decoder (T5-style), designed for seq2seq tasks
#   - Fine-tuning updates all weights for your specific task+domain
#   - Even 4799 examples is enough to fine-tune a pretrained seq2seq model

# Expected ROUGE-1: 15-30 (vs best LSTM = 5.31)
# Training time: ~3-4 hours on RTX 3050

# Usage:
#     pip install transformers datasets sentencepiece
#     python src/train_mt5.py
#     python src/train_mt5.py --model google/mt5-small --epochs 10
# """

# """
# Memory-optimized mT5 fine-tuning for 4GB GPU
# """
# """
# src/train_mt5.py  — Fine-tune mT5-small for Malayalam summarization
# """

# import os
# os.environ["TRANSFORMERS_NO_TF"]    = "1"
# os.environ["TRANSFORMERS_NO_FLAX"]  = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import sys, argparse, pickle
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
#                           Seq2SeqTrainer, Seq2SeqTrainingArguments,
#                           DataCollatorForSeq2Seq, EarlyStoppingCallback)

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config
# from src.evaluate import compute_rouge


# class MalayalamSumDataset(Dataset):
#     def __init__(self, samples, tokenizer, max_src=64, max_tgt=24):
#         self.tokenizer = tokenizer
#         self.max_src   = max_src
#         self.max_tgt   = max_tgt
#         self.data = [
#             {'src': ' '.join(s['src_tokens']),
#              'tgt': ' '.join(s['tgt_tokens'])}
#             for s in samples
#             if s['src_tokens'] and s['tgt_tokens']
#         ]
#         print(f"  Dataset: {len(self.data)} samples")

#     def __len__(self): return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         enc  = self.tokenizer("summarize: " + item['src'],
#                               max_length=self.max_src, truncation=True)
#         lab  = self.tokenizer(text_target=item['tgt'],
#                               max_length=self.max_tgt, truncation=True)
#         enc['labels'] = lab['input_ids']
#         return enc


# def compute_metrics_fn(tokenizer):
#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#         if isinstance(preds, tuple): preds = preds[0]
#         preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         dp = tokenizer.batch_decode(preds,  skip_special_tokens=True)
#         dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         r1, r2, rl = [], [], []
#         for p, r in zip(dp, dl):
#             if p.strip() and r.strip():
#                 s = compute_rouge(r, p)
#                 r1.append(s['rouge1']); r2.append(s['rouge2']); rl.append(s['rougeL'])
#         n = max(len(r1), 1)
#         return {'rouge1': round(sum(r1)/n, 2),
#                 'rouge2': round(sum(r2)/n, 2),
#                 'rougeL': round(sum(rl)/n, 2)}
#     return compute_metrics


# def train(model_name='google/mt5-small', epochs=15, batch_size=1):
#     print(f"\n{'='*60}")
#     print(f"Fine-tuning: {model_name} | epochs={epochs} | device={config.DEVICE}")
#     print(f"{'='*60}")

#     with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'rb') as f:
#         train_data = pickle.load(f)
#     with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'),  'rb') as f:
#         test_data  = pickle.load(f)

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     model.gradient_checkpointing_enable()
#     print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

#     train_ds = MalayalamSumDataset(train_data, tokenizer)
#     test_ds  = MalayalamSumDataset(test_data,  tokenizer)
#     collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

#     output_dir = os.path.join(config.CHECKPOINTS_DIR, 'mt5')
#     os.makedirs(output_dir, exist_ok=True)

#     args = Seq2SeqTrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=5,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=4,
#         learning_rate=1e-4,
#         warmup_steps=50,
#         lr_scheduler_type="linear",
#         max_grad_norm=1.0,
#         weight_decay=0.01,
#         fp16=True,
#         predict_with_generate=True,
#         generation_max_length=32,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=1,
#         load_best_model_at_end=True,
#         metric_for_best_model="rouge1",
#         greater_is_better=True,
#         logging_steps=100,
#         report_to="none",
#         dataloader_pin_memory=False,
#         optim="adafactor",
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=test_ds,
#         data_collator=collator,
#         compute_metrics=compute_metrics_fn(tokenizer),
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
#     )

#     print("\nStarting fine-tuning...\n")
#     trainer.train()

#     results = trainer.evaluate()
#     r1 = results.get('eval_rouge1', 0)
#     r2 = results.get('eval_rouge2', 0)
#     rl = results.get('eval_rougeL', 0)
#     print(f"\n{'='*40}")
#     print(f"ROUGE-1={r1:.2f} | ROUGE-2={r2:.2f} | ROUGE-L={rl:.2f}")
#     print(f"{'='*40}")

#     best = os.path.join(output_dir, 'best')
#     trainer.save_model(best)
#     tokenizer.save_pretrained(best)
#     print(f"Saved → {best}")
#     return results


# def generate_summary(text, model_path=None):
#     if model_path is None:
#         model_path = os.path.join(config.CHECKPOINTS_DIR, 'mt5', 'best')
#     tok   = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     model.eval()
#     inp = tok("summarize: " + text, return_tensors='pt',
#               max_length=128, truncation=True)
#     with torch.no_grad():
#         out = model.generate(**inp, max_new_tokens=64,
#                              num_beams=4, no_repeat_ngram_size=3)
#     return tok.decode(out[0], skip_special_tokens=True)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model',  default='google/mt5-small')
#     parser.add_argument('--epochs', type=int, default=15)
#     parser.add_argument('--batch',  type=int, default=1)
#     parser.add_argument('--infer',  type=str, default=None)
#     args = parser.parse_args()
#     if args.infer:
#         print(generate_summary(args.infer))
#     else:
#         train(args.model, args.epochs, args.batch)


# """
# src/train_mt5.py — Fine-tune mT5-small for Malayalam summarization
# Fixes: removed adafactor (causes NaN with fp16), use AdamW instead
#        removed fp16 (causes NaN with mT5 — use bf16 on Colab A100/T4)
#        fixed learning_rate schedule
# """

# import os, sys, argparse, pickle
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
#                           Seq2SeqTrainer, Seq2SeqTrainingArguments,
#                           DataCollatorForSeq2Seq, EarlyStoppingCallback)

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import config
# from src.evaluate import compute_rouge


# class MalayalamSumDataset(Dataset):
#     def __init__(self, samples, tokenizer, max_src=128, max_tgt=40):
#         self.tokenizer = tokenizer
#         self.data = [
#             {'src': ' '.join(s['src_tokens']),
#              'tgt': ' '.join(s['tgt_tokens'])}
#             for s in samples
#             if s['src_tokens'] and s['tgt_tokens']
#         ]
#         self.max_src = max_src
#         self.max_tgt = max_tgt
#         print(f"  Dataset: {len(self.data)} samples")

#     def __len__(self): return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         enc = self.tokenizer(
#             "summarize: " + item['src'],
#             max_length=self.max_src, truncation=True
#         )
#         lab = self.tokenizer(
#             text_target=item['tgt'],
#             max_length=self.max_tgt, truncation=True
#         )
#         enc['labels'] = lab['input_ids']
#         return enc


# def compute_metrics_fn(tokenizer):
#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#         if isinstance(preds, tuple): preds = preds[0]
#         # clip to valid token range to avoid decode errors
#         vocab_size = tokenizer.vocab_size
#         preds  = np.clip(np.where(preds  != -100, preds,  tokenizer.pad_token_id), 0, vocab_size-1)
#         labels = np.clip(np.where(labels != -100, labels, tokenizer.pad_token_id), 0, vocab_size-1)
#         dp = tokenizer.batch_decode(preds,  skip_special_tokens=True)
#         dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         r1, r2, rl = [], [], []
#         for p, r in zip(dp, dl):
#             if p.strip() and r.strip():
#                 s = compute_rouge(r, p)
#                 r1.append(s['rouge1'])
#                 r2.append(s['rouge2'])
#                 rl.append(s['rougeL'])
#         n = max(len(r1), 1)
#         # Print a few samples so you can see actual output
#         for i in range(min(2, len(dp))):
#             print(f"\n  REF: {dl[i][:80]}")
#             print(f"  HYP: {dp[i][:80]}")
#         return {
#             'rouge1': round(sum(r1)/n, 2),
#             'rouge2': round(sum(r2)/n, 2),
#             'rougeL': round(sum(rl)/n, 2),
#         }
#     return compute_metrics


# def train(model_name='google/mt5-small', epochs=10, batch_size=4):
#     print(f"\n{'='*60}")
#     print(f"Fine-tuning: {model_name}")
#     print(f"epochs={epochs} | batch={batch_size} | device={config.DEVICE}")
#     print(f"{'='*60}")

#     # Detect GPU type for precision setting
#     use_bf16 = False
#     use_fp16 = False
#     if torch.cuda.is_available():
#         gpu_name = torch.cuda.get_device_name(0)
#         print(f"GPU: {gpu_name}")
#         # bf16 works on Ampere+ (A100, RTX 30xx, RTX 40xx)
#         # fp16 causes NaN with mT5 — avoid it
#         if torch.cuda.is_bf16_supported():
#             use_bf16 = True
#             print("Using bf16 (stable for mT5)")
#         else:
#             print("bf16 not supported — using fp32 (slower but stable)")

#     with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'rb') as f:
#         train_data = pickle.load(f)
#     with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'), 'rb') as f:
#         test_data = pickle.load(f)

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Gradient checkpointing saves memory — disable use_cache which conflicts
#     model.config.use_cache = False
#     model.gradient_checkpointing_enable()

#     n = sum(p.numel() for p in model.parameters())
#     print(f"Params: {n/1e6:.0f}M")

#     train_ds = MalayalamSumDataset(train_data, tokenizer)
#     test_ds  = MalayalamSumDataset(test_data,  tokenizer)
#     collator = DataCollatorForSeq2Seq(tokenizer, model=model,
#                                       padding=True, pad_to_multiple_of=8)

#     output_dir = os.path.join(config.CHECKPOINTS_DIR, 'mt5')
#     os.makedirs(output_dir, exist_ok=True)

#     # Effective batch = batch_size * gradient_accumulation_steps
#     # On Colab T4: batch=4 * accum=4 = 16
#     # On local 4GB: batch=1 * accum=16 = 16
#     accum = max(1, 16 // batch_size)

#     args = Seq2SeqTrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         gradient_accumulation_steps=accum,

#         # AdamW — stable, works with bf16/fp32, NOT adafactor
#         # adafactor causes NaN gradients with fp16
#         optim='adamw_torch',
#         learning_rate=5e-4,
#         lr_scheduler_type='cosine',
#         warmup_ratio=0.05,
#         weight_decay=0.01,
#         max_grad_norm=1.0,

#         bf16=use_bf16,
#         fp16=False,           # NEVER use fp16 with mT5 — causes NaN

#         predict_with_generate=True,
#         generation_max_length=64,
#         generation_num_beams=4,

#         eval_strategy='epoch',
#         save_strategy='epoch',
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         metric_for_best_model='rouge1',
#         greater_is_better=True,

#         logging_steps=50,
#         report_to='none',
#         dataloader_pin_memory=False,
#         dataloader_num_workers=0,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=test_ds,
#         tokenizer=tokenizer,
#         data_collator=collator,
#         compute_metrics=compute_metrics_fn(tokenizer),
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
#     )

#     print(f"\nEffective batch size: {batch_size * accum}")
#     print("Starting fine-tuning...\n")
#     trainer.train()

#     results = trainer.evaluate()
#     r1 = results.get('eval_rouge1', 0)
#     r2 = results.get('eval_rouge2', 0)
#     rl = results.get('eval_rougeL', 0)
#     print(f"\n{'='*50}")
#     print(f"  mT5-small FINAL: ROUGE-1={r1:.2f} | ROUGE-2={r2:.2f} | ROUGE-L={rl:.2f}")
#     print(f"{'='*50}")

#     best = os.path.join(output_dir, 'best')
#     trainer.save_model(best)
#     tokenizer.save_pretrained(best)
#     print(f"Saved → {best}")
#     return results


# def generate_summary(text, model_path=None):
#     if model_path is None:
#         model_path = os.path.join(config.CHECKPOINTS_DIR, 'mt5', 'best')
#     tok   = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     model.eval()
#     inp = tok("summarize: " + text, return_tensors='pt',
#               max_length=128, truncation=True)
#     with torch.no_grad():
#         out = model.generate(
#             **inp, max_new_tokens=64,
#             num_beams=4, no_repeat_ngram_size=3,
#             early_stopping=True,
#         )
#     return tok.decode(out[0], skip_special_tokens=True)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model',  default='google/mt5-small')
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch',  type=int, default=4)
#     parser.add_argument('--infer',  type=str, default=None)
#     args = parser.parse_args()
#     if args.infer:
#         print(generate_summary(args.infer))
#     else:
#         train(args.model, args.epochs, args.batch)


#save in colab drive
"""
src/train_mt5.py — Fine-tune mT5-small for Malayalam summarization
With Google Drive saving support for Colab
"""

import os, sys, argparse, pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.evaluate import compute_rouge


class MalayalamSumDataset(Dataset):
    def __init__(self, samples, tokenizer, max_src=128, max_tgt=40):
        self.tokenizer = tokenizer
        self.data = [
            {'src': ' '.join(s['src_tokens']),
             'tgt': ' '.join(s['tgt_tokens'])}
            for s in samples
            if s['src_tokens'] and s['tgt_tokens']
        ]
        self.max_src = max_src
        self.max_tgt = max_tgt
        print(f"  Dataset: {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer("summarize: " + item['src'],
                             max_length=self.max_src, truncation=True)
        lab = self.tokenizer(text_target=item['tgt'],
                             max_length=self.max_tgt, truncation=True)
        enc['labels'] = lab['input_ids']
        return enc


def compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        vocab_size = tokenizer.vocab_size
        preds  = np.clip(np.where(preds  != -100, preds,  tokenizer.pad_token_id), 0, vocab_size-1)
        labels = np.clip(np.where(labels != -100, labels, tokenizer.pad_token_id), 0, vocab_size-1)
        dp = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        r1, r2, rl = [], [], []
        for p, r in zip(dp, dl):
            if p.strip() and r.strip():
                s = compute_rouge(r, p)
                r1.append(s['rouge1']); r2.append(s['rouge2']); rl.append(s['rougeL'])
        n = max(len(r1), 1)
        print(f"\n  Sample — REF: {dl[0][:70] if dl else ''}")
        print(f"           HYP: {dp[0][:70] if dp else ''}")
        return {
            'rouge1': round(sum(r1)/n, 2),
            'rouge2': round(sum(r2)/n, 2),
            'rougeL': round(sum(rl)/n, 2),
        }
    return compute_metrics


def train(model_name='google/mt5-small', epochs=12, batch_size=4,
          drive_save_path=None, resume_from=None):
    """
    Args:
        drive_save_path: if set, copies best checkpoint to Google Drive after each epoch
                         e.g. '/content/drive/MyDrive/mt5_malayalam'
        resume_from    : path to resume training from a checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"epochs={epochs} | batch={batch_size} | device={config.DEVICE}")
    if drive_save_path:
        print(f"Drive save: {drive_save_path}")
    print(f"{'='*60}")

    # Detect precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Precision: {'bf16' if use_bf16 else 'fp32'}")

    with open(os.path.join(config.DATA_PROCESSED, 'train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(config.DATA_PROCESSED, 'test.pkl'), 'rb') as f:
        test_data  = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Resume from checkpoint or load fresh
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    train_ds = MalayalamSumDataset(train_data, tokenizer)
    test_ds  = MalayalamSumDataset(test_data,  tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                      padding=True, pad_to_multiple_of=8)

    output_dir = os.path.join(config.CHECKPOINTS_DIR, 'mt5')
    os.makedirs(output_dir, exist_ok=True)

    accum = max(1, 16 // batch_size)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        optim='adamw_torch',
        learning_rate=5e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=4,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,            # keep last 3 checkpoints in output_dir
        load_best_model_at_end=True,
        metric_for_best_model='rouge1',
        greater_is_better=True,
        logging_steps=50,
        report_to='none',
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    # Custom callback to copy best model to Google Drive after each epoch
    if drive_save_path:
        import shutil
        from transformers import TrainerCallback

        class DriveCallback(TrainerCallback):
            def __init__(self, drive_path, tok):
                self.drive_path = drive_path
                self.tok = tok
                self.best_rouge = 0.0

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                r1 = metrics.get('eval_rouge1', 0) if metrics else 0
                if r1 > self.best_rouge:
                    self.best_rouge = r1
                    os.makedirs(self.drive_path, exist_ok=True)
                    # Save model to drive
                    trainer.model.save_pretrained(self.drive_path)
                    self.tok.save_pretrained(self.drive_path)
                    # Save metrics
                    with open(os.path.join(self.drive_path, 'metrics.txt'), 'w') as f:
                        f.write(f"epoch={state.epoch:.1f}\n")
                        f.write(f"rouge1={r1:.2f}\n")
                        if metrics:
                            for k, v in metrics.items():
                                f.write(f"{k}={v}\n")
                    print(f"\n  Saved to Drive: {self.drive_path} (ROUGE-1={r1:.2f})")

        drive_callback = DriveCallback(drive_save_path, tokenizer)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=4), drive_callback]
    else:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=callbacks,
    )

    print(f"\nEffective batch size: {batch_size * accum}")
    print("Starting fine-tuning...")
    print("Checkpoints saved every epoch to:", output_dir)
    if drive_save_path:
        print("Best model also saved to Drive:", drive_save_path)
    print()

    # Resume training from checkpoint if directory has checkpoints
    resume_ckpt = None
    if resume_from and os.path.exists(resume_from):
        resume_ckpt = resume_from

    trainer.train(resume_from_checkpoint=resume_ckpt)

    results = trainer.evaluate()
    r1 = results.get('eval_rouge1', 0)
    r2 = results.get('eval_rouge2', 0)
    rl = results.get('eval_rougeL', 0)

    print(f"\n{'='*50}")
    print(f"mT5-small FINAL: ROUGE-1={r1:.2f} | ROUGE-2={r2:.2f} | ROUGE-L={rl:.2f}")
    print(f"{'='*50}")

    # Save final best model
    best_local = os.path.join(output_dir, 'best')
    trainer.save_model(best_local)
    tokenizer.save_pretrained(best_local)
    print(f"Saved locally → {best_local}")

    if drive_save_path:
        import shutil
        trainer.model.save_pretrained(drive_save_path)
        tokenizer.save_pretrained(drive_save_path)
        print(f"Saved to Drive → {drive_save_path}")

    return results


def generate_summary(text, model_path=None):
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINTS_DIR, 'mt5', 'best')
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    inp = tok("summarize: " + text, return_tensors='pt',
              max_length=128, truncation=True)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=64,
            num_beams=4, no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='google/mt5-small')
    parser.add_argument('--epochs',  type=int, default=12)
    parser.add_argument('--batch',   type=int, default=4)
    parser.add_argument('--drive',   type=str, default=None,
                        help='Google Drive path, e.g. /content/drive/MyDrive/mt5_malayalam')
    parser.add_argument('--resume',  type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--infer',   type=str, default=None)
    args = parser.parse_args()

    if args.infer:
        print(generate_summary(args.infer))
    else:
        train(args.model, args.epochs, args.batch,
              drive_save_path=args.drive,
              resume_from=args.resume)