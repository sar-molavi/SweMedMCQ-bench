# Refinement of the test translations

The refinement loop checks every GPT-5.4 draft against its English source and repairs it conservatively. The code is in `swemedmcq/refine/`, the configuration of the released run in `configs/refine/with_gemma.yaml`, and the prompts in `swemedmcq/refine/prompts/`.

## The loop

1. **Normalisation (code).** L0 fixes typography (Unicode NFC, spaces, hyphens, control characters) and rebuilds the MedXpertQA answer list inside the question from the Swedish options (`normalise.py`, `medxpertqa.py`). L1 applies Swedish number style (decimal comma, thousands space) with the English text as reference (`numbers.py`, `locale.py`).
2. **Passes.** Each item runs up to three rounds of three passes: *meaning*, *terminology*, *language* (`loop/item_loop.py`). In each pass:
   - **detect**: GPT-OSS-120B (and, for meaning, also Qwen3.8-27B) lists possible errors with a quoted span;
   - **ground**: code checks each claim against the Swedish sources in `data/resources/` and drops claims against valid Swedish (`termbase.py`, `loop/routing.py`);
   - **verify**: a sceptical prompt confirms or rejects the claim; decoy findings check that the verifier does not confirm everything;
   - **repair**: code (Swedish drug names from the national product register NPL or Svensk MeSH, missing units) or GPT-OSS writes a candidate edit;
   - **gate**: deterministic checks reject edits that lose a number, add a non-Swedish word, change the length too much, and so on (`gate.py`).
3. **Vote.** Blind pairwise judges compare the current text with the candidate, in both orders: Qwen3.8-27B three samples per order, GPT-OSS-120B and Gemma-4-31B one each (10 votes). With `c` the share of votes for the candidate, an edit is accepted with *high* confidence if `c ≥ 0.9` and no judge prefers the current text, and with *medium* confidence if `c ≥ 0.75` with at most one such veto, not from the model family that wrote the edit (`loop/accept.py`).
4. **Tiers.** Edits made by code form tier 1; edits written by a model form tier 2. The benchmark uses the tier-1 layer; the review ([REVIEW_PROTOCOL](REVIEW_PROTOCOL.md)) explains why.

Model settings (`configs/refine/with_gemma.yaml`): vLLM, 32,768-token context, up to 16,384 output tokens; GPT-OSS-120B greedy at reasoning effort medium; Qwen3.8-27B with thinking at its default effort, temperature 1.0, top-p 0.95, top-k 20; Gemma-4-31B with native thinking, temperature 1.0, top-p 0.95, top-k 64. A failed GPT-OSS call is retried once at temperature 0.6, then sent to Qwen. A field is retranslated only for a critical meaning finding whose repair was rejected, in an option or a field of at most 60 words, or when the draft field is corrupt.

## Running it

```bash
bash scripts/refine/fetch_offline.sh /shared/tiktoken && export TIKTOKEN_ENCODINGS_BASE=/shared/tiktoken  # offline nodes
RUN=full SHARDS=8 STAGES=prepare bash scripts/03_refine.sh     # CPU: L0, L1, shards
RUN=full SHARD=0 STAGES=shard bash scripts/03_refine.sh         # GPU: one shard (repeat for 0..7)
RUN=full STAGES=probes bash scripts/03_refine.sh                # GPU: judge probes
RUN=full STAGES="assemble expert" bash scripts/03_refine.sh     # CPU: release layers, review sheets
```

Each shard runs on one node with 4 × H200 (141 GB): GPT-OSS on one GPU, Qwen on two, Gemma on one. On Slurm, `scripts/slurm/refine_shards.sbatch` submits the eight shards as an array. `scripts/refine/engine_smoke.sh` loads all models once and checks the outputs before a full run. `configs/refine/default.yaml` is the same loop without Gemma.

Inputs: the draft (`configs/paths.yaml` `draft`, the released one is `data/test/layers/layer1_draft.jsonl`), the gold letters (`benchmark`) and the term base (`resources`).

Outputs in `runs/refine/<RUN>/`:

| Folder | Content |
|---|---|
| `prep/` | the normalised items, one file per shard |
| `shards/<k>/results/` | one result per item |
| `release/` | `layer1_draft`, `layer2_hygiene`, `layer3_locale`, `layer4a_auto` (tier 1), `layer4_refined` (tiers 1+2), `provenance.jsonl`, `report.json`, `expert_queue.jsonl` |
| `probes/` | judge probes and their results |
| `expert/` | blinded review sheets and their key (`key.json`) |

## From the run to the benchmark

```bash
RUN=full ANSWERS=<gold letters, JSON or JSONL> bash scripts/04_build_testset.sh
```

writes `data/test/benchmark.jsonl` (layer 4a, `release/layer4a_auto.jsonl`, the final benchmark), the other layers in `data/test/layers/`, `expert_queue.jsonl`, `edit_history.jsonl.gz` (the provenance of every item) and `refinement_report.json`, and then the evaluation prompt files. The `hard` flag comes from the `test_hard.jsonl` files of the source benchmarks. For the released run, this reproduces the released files byte for byte. Without `RUN`, the script only builds the prompt files from the released benchmark.

## Judge probes

`python -m swemedmcq.refine.evaluation.probes --run-dir runs/refine/full --config configs/refine/with_gemma.yaml` checks each judge on pairs with a known answer: 250 pairs with one injected error (English drug name, number, agreement, negation, laterality), 50 identical pairs and 40 labelled grammar edits (`data/resources/grammar_edit_sample_labels.csv`), each in both orders. The released results are in `data/test/probes/`.

## Term base

`data/resources/` holds the built term base and word lists the loop reads (sources and licences in [`data/README.md`](../data/README.md#term-base-and-swedish-resources)). `bash scripts/refine/build_resources.sh` rebuilds them from the raw sources (Svensk MeSH and Wikidata are downloaded; SALDO/SALDOM, NPL, KSH97-P, hunspell, the Socialstyrelsen termbank, KVÅ and FreeDict must be placed in `work/termbase_raw` first, see `swemedmcq/refine/resources/build_termbase.py`).

`python -m swemedmcq.refine.check_medxpertqa_tails` checks the rebuilt MedXpertQA answer lists of a layer against its options.
