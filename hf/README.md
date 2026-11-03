# Released models

The 14 students are on the Hugging Face Hub as `Anonymiserad/<name>`:

| Base | Answer-level | Reasoning trace |
|---|---|---|
| LFM2.5-1.2B | `SweMedMCQ-LFM2.5-1.2B-Answer-{EN,SV,EN-SV}` | `SweMedMCQ-LFM2.5-1.2B-Trace-{EN,SV,SV-RW,EN-SV}` |
| EuroLLM-1.7B | `SweMedMCQ-EuroLLM-1.7B-Answer-{EN,SV,EN-SV}` | `SweMedMCQ-EuroLLM-1.7B-Trace-{EN,SV,SV-RW,EN-SV}` |

Each repository holds:
- the checkpoint with the best validation loss (`model.safetensors`, bfloat16);
- the config and the tokenizer and chat template it was evaluated with;
- `prompts.json`, the English and Swedish prompts;
- the licence;
- a model card covering training data, settings, prompt format, vLLM usage and results.

## Running a model with vLLM

The evaluation used greedy decoding, the prompts in `prompts.json` (one user message, options as `A) …` separated by blank lines), and these settings:

| Students | Settings |
|---|---|
| LFM2.5 | `max_model_len=12288`. The trace students' chat template opens the reasoning block; pass `enable_thinking=False` for a direct answer. |
| EuroLLM | `max_model_len=12288`, `hf_overrides={"max_position_embeddings": 12288}`. Trace students: append `<think>\n` to the generation prompt to reason, or `<think>\n\n</think>\n\n` for a direct answer. |
| Answer-level | JSON-schema decoding `{"answer_index": <letter>}` from the first token |

```python
import json
import re

from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

MODEL = "Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Trace-EN-SV"
PROMPTS = json.load(open(hf_hub_download(MODEL, "prompts.json"), encoding="utf-8"))
llm = LLM(MODEL, max_model_len=12288)
user = PROMPTS["sv"].format(question="Vilket vitamin saknas vid skörbjugg?",
                            options="A) A-vitamin\n\nB) C-vitamin\n\nC) D-vitamin")
prompt = llm.get_tokenizer().apply_chat_template([{"role": "user", "content": user}],
                                                 add_generation_prompt=True, tokenize=False)
text = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=12288))[0].outputs[0].text
match = re.search(r'"answer_index"\s*:\s*"\s*([A-Z])', text.split("</think>")[-1])
print(match.group(1) if match else None)  # None: no valid answer
del llm  # shut the engine down, so the script exits
```

Serving: `vllm serve Anonymiserad/<name> --max-model-len 12288` (EuroLLM: add `--hf-overrides '{"max_position_embeddings": 12288}'`), then query it with the OpenAI client; each model card has the exact request. To run the full benchmark on the students, see [`docs/EVALUATION.md`](../docs/EVALUATION.md).

## Upload

`models/` (not tracked in git) holds the local copies that were uploaded. To publish them, or a newly trained student folder:

```bash
hf auth login
HF_USER=<user or organisation> bash hf/upload.sh [folder ...]
```

The script creates each public repository, uploads the folder with `hf upload`, and adds the models to a collection.
