# Training the students

Each of the 14 students has one config in `configs/train/`, named like its released model (`SweMedMCQ-<Base>-<Answer|Trace>-<EN|SV|SV-RW|EN-SV>.json`). The config both tokenises the data (`swemedmcq.train.preprocess_dataset -c`, the `preprocess` step of [DISTILLATION](DISTILLATION.md)) and trains.

```bash
# training environment (requirements/train.txt), 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/07_train.sh SweMedMCQ-LFM2.5-1.2B-Trace-EN-SV
DRY_RUN=1 bash scripts/07_train.sh SweMedMCQ-EuroLLM-1.7B-Trace-SV      # checks + launch command only
mkdir -p logs && sbatch --export=ALL,STUDENT=SweMedMCQ-EuroLLM-1.7B-Trace-SV,ENV_SETUP=<file> scripts/slurm/train.sbatch
```

`07_train.sh`:

1. Runs `swemedmcq.train.check_config`, a CPU dry run. It checks every input path, the tokenised data against its manifest (row counts, lengths, labels), that the trace data comes only from the two teachers, and prints the schedule.
2. Launches `accelerate` with DeepSpeed ZeRO-2 on 4 GPUs (`configs/train/accelerate_deepspeed_zero2_4gpu.yaml`, `deepspeed_zero2.json`). It uses `swemedmcq.train.train_unsloth_full`, or `swemedmcq.train.bilingual.train_unsloth_full` for the EN+SV students, which evaluate on the English and the Swedish validation sets.
3. Post-step: EuroLLM checkpoints get the complete tokenizer files (`finalize_checkpoint`). The LFM trace checkpoints get the reasoning chat template (`create_lfm_dynamic_tokenizer`), whose `enable_thinking` switch opens `<think>` or writes an empty block.

A crashed or timed-out run continues from its newest checkpoint when the same command runs again. The final model is `runs/train/<student>/final_checkpoint`, the checkpoint with the best validation loss.

## Settings (all students)

| Setting | Value |
|---|---|
| Method | full fine-tuning (Unsloth + TRL), embeddings trained, bfloat16 |
| Base models | `LiquidAI/LFM2.5-1.2B-Instruct` (most configs load its `unsloth/` mirror, which has the same weights), `utter-project/EuroLLM-1.7B-Instruct` |
| Epochs | 2 |
| Batch | 32 per GPU × 4 GPUs = 128 sequences, padding-free packing |
| Optimiser | learning rate 5e-5, cosine schedule, 5% warm-up, weight decay 0.01, gradient clipping 1.0 |
| Loss | negative log-likelihood on the response tokens |
| Sequence length | EuroLLM 12,288; LFM2.5 its full context |
| Validation | every 200 steps on the train-time validation set; early stopping with patience 4 (trace students) or 10 (answer-level students; 2 for LFM2.5 Answer-SV and EuroLLM Answer-EN-SV); best checkpoint kept |
| Seed | 13 |

The answer-level configs set `answer_level_sft: true`: the target is only the JSON answer `{"answer_index": "<letter>"}`.

## Using a trained model

Point the student's `model` in `configs/models.yaml` to `runs/train/<student>/final_checkpoint`, or copy the folder to `hf/models/<student>`, and evaluate it ([EVALUATION](EVALUATION.md)). `bash hf/upload.sh <folder>` publishes a folder on the Hugging Face Hub.
