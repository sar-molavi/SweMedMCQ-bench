# Scripts

One runbook per stage. Run them from the repository root. Each stage script (`01`–`08`) resumes where it stopped, runs only some of its steps with `STAGES="a b"`, and prints its commands without running them with `DRY_RUN=1`. The header of each script lists its steps, inputs, outputs and environment variables. [`docs/PIPELINE.md`](../docs/PIPELINE.md) shows the order.

| Script | Stage | Needs |
|---|---|---|
| `01_translate_test.sh` | GPT-5.4 drafts of the test items | OpenAI Batch API (paid) |
| `02_translate_train.sh` | GPT-OSS-120B translation of the training questions | GPUs |
| `03_refine.sh` | refinement loop, judge probes, release layers, review sheets | 4 × 141 GB GPUs per shard |
| `04_build_testset.sh` | benchmark files (from a refinement run) and evaluation prompt files | CPU |
| `05_teacher_traces.sh` | teacher traces in English and Swedish | GPUs |
| `06_build_sft_data.sh` | training data, Swedish rewrite, answer-level data, tokenisation | CPU (GPUs for the rewrite) |
| `07_train.sh <student>` | fine-tuning of one student | 4 GPUs, training environment |
| `08_evaluate.sh` | evaluation jobs, scoring, per-item results, paired statistics | GPUs |
| `common.sh` | shared helpers (`PYTHON`, `STAGES`, `DRY_RUN`, paths) | – |
| `refine/build_resources.sh` | rebuild the term base and word lists | raw sources |
| `refine/fetch_offline.sh` | vocab files for GPT-OSS on offline nodes | internet |
| `refine/engine_smoke.sh` | load the refinement models once and check their outputs | GPUs |
| `slurm/` | Slurm examples: refinement shards, one student, one evaluation job | Slurm |
