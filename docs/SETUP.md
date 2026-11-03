# Setup

## Environments

Two Python 3.12 environments are used: one for everything except fine-tuning, one for fine-tuning.

| Environment | File | Main packages | Used for |
|---|---|---|---|
| Inference | `requirements/inference.txt` | vLLM 0.28.0, torch 2.13, transformers 5.16.1, xgrammar, openai-harmony | translation, refinement, test-set build, teacher traces, SFT data and tokenisation, evaluation, review scoring, tests |
| Training | `requirements/train.txt` | Unsloth 2026.8.22, TRL 0.24.0, transformers 5.5.0, torch 2.11 (CUDA 13), DeepSpeed 0.19.6, accelerate | fine-tuning (`scripts/07_train.sh`) |

```bash
python3.12 -m venv ~/envs/swemedmcq && source ~/envs/swemedmcq/bin/activate
pip install -r requirements/inference.txt
pip install -e .            # makes `python -m swemedmcq...` work from anywhere
```

The same for the training environment, with `requirements/train.txt`. The scripts use `python` of the active environment; set `PYTHON=/path/to/python` to use another one. They also put the repository root on `PYTHONPATH`, so `pip install -e .` is optional for them.

Check the installation with the unit tests (inference environment, CPU only, about a minute):

```bash
python -m pytest tests
```

## Hardware

| Stage | Hardware used for the released runs |
|---|---|
| Test-set drafts (GPT-5.4) | OpenAI Batch API (paid) |
| Training translation, teacher traces, Swedish rewrite | GPUs with room for GPT-OSS-120B (about 65 GB of weights) or Qwen3.6-35B-A3B (about 70 GB); tensor parallelism with `CUDA_VISIBLE_DEVICES` |
| Refinement loop | one node with 4 × H200 (141 GB) per shard; 8 shards |
| Fine-tuning | 4 GPUs per student (DeepSpeed ZeRO-2) |
| Evaluation | 1 GPU per job (2 for GPT-OSS-120B) |
| Everything else | CPU |

## Data that is not in the repository

**English source benchmarks.** Translation, the `hard` flag and the training pipeline read the English test and training questions of MedicalAgentsBench ([arXiv:2503.07459](https://arxiv.org/abs/2503.07459)), one folder per dataset:

```
data/source/<dataset>/train.jsonl       training questions (stages 1b, 4)
data/source/<dataset>/test.jsonl        test questions (stage 1a)
data/source/<dataset>/test_hard.jsonl   hard subset of the test questions (the `hard` flag)
```

with the datasets `afrimedqa`, `medbullets`, `medexqa`, `medmcqa`, `medqa`, `medxpertqa-r`, `medxpertqa-u`, `mmlu`, `mmlu-pro` and `pubmedqa`, and the fields `realidx`, `question`, `options` and `answer_idx`. Use the version at commit [`7b9ecfc`](https://github.com/gersteinlab/MedicalAgentsBench/tree/7b9ecfc) (10 April 2025) of the MedicalAgentsBench repository: its authors later corrected and removed some items, and the benchmark and splits here follow the earlier version. Put the dataset folders of its `data/` folder there (or point `source_benchmark` in `configs/paths.yaml` to them). The benchmark itself (`data/test/benchmark.jsonl`) already contains the English test questions, so **using or evaluating on the benchmark needs no source files**.

**Swedish training translation.** Stage 1b (`scripts/02_translate_train.sh`) writes it to `data/train/sv_train_translation.jsonl`; it is the input of the Swedish training data.

**Teacher traces and training data** are not shipped; stages 4a–4b rebuild them. The train/validation split of the released students is fixed by `data/train/split/` (see [DISTILLATION](DISTILLATION.md)).

## Paths

Every location is in `configs/paths.yaml`, relative to the repository root. Override one with an environment variable `SWEMEDMCQ_<KEY>` (upper case), or move the whole root with `SWEMEDMCQ_ROOT`:

```bash
python -m swemedmcq.paths                 # print every location
python -m swemedmcq.paths results_dir     # print one
SWEMEDMCQ_WORK_DIR=/scratch/swemedmcq bash scripts/06_build_sft_data.sh
```

Outputs go to `work/` (intermediate training data) and `runs/` (refinement, test-set, training and evaluation runs); both are ignored by git. Models are read from the Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`).

## Offline GPU nodes

- Download the models into the Hugging Face cache on a node with internet, then set `HF_HUB_OFFLINE=1`.
- GPT-OSS needs the `openai_harmony` vocab files, which it otherwise downloads at run time:

  ```bash
  bash scripts/refine/fetch_offline.sh /shared/tiktoken
  export TIKTOKEN_ENCODINGS_BASE=/shared/tiktoken
  ```

## Slurm

`scripts/slurm/` has examples for the GPU stages: `refine_shards.sbatch` (one 4-GPU job per refinement shard), `train.sbatch` (one student) and `eval_job.sh` (one evaluation job; `swemedmcq.evaluate.make_jobs` also writes array files). Replace the placeholders (`<ACCOUNT>`, `<GPU_PARTITION>`, `<CLUSTER_DIR>`) and set `ENV_SETUP` to a file that activates the environment. Create `logs/` before submitting.
