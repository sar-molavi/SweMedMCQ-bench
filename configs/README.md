# Configs

| File | Used by | Content |
|---|---|---|
| `paths.yaml` | every stage (`swemedmcq.paths`) | every input and output location, relative to the repository root; override one with `SWEMEDMCQ_<KEY>` |
| `models.yaml` | evaluation (`scripts/08_evaluate.sh`) | every evaluated configuration: model, prompt, reasoning mode, parser, generation settings, test sets; the header explains each field |
| `refine/with_gemma.yaml` | refinement (`scripts/03_refine.sh`) | the configuration of the released run: GPT-OSS-120B, Qwen3.8-27B and Gemma-4-31B, their GPUs and sampling, retries, loop settings (passes, rounds, judges, acceptance thresholds) |
| `refine/default.yaml` | refinement | the same loop with GPT-OSS-120B and Qwen3.8-27B only |
| `train/SweMedMCQ-*.json` | training (`scripts/07_train.sh`) and tokenisation | one config per released student: base model, data paths, hyperparameters |
| `train/preprocess/` | tokenisation | the tokenisation config of the LFM2.5 answer-level EN+SV data (tokenised from the mixed file) |
| `train/accelerate_deepspeed_zero2_4gpu.yaml`, `train/deepspeed_zero2.json` | training | 4-GPU launch with DeepSpeed ZeRO-2 |
