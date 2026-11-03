"""Training helpers: GPU-memory logging, VRAM reservation and the model's maximum sequence length."""

import json
import os
import re

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PretrainedConfig,
)
import torch


class GPUMemoryTrackerCallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback that logs GPU memory usage.

    This callback hooks into the training loop and prints detailed GPU memory
    statistics (allocated and reserved) every `log_every_n_steps`. This is
    particularly useful for:
    - Monitoring memory consumption to prevent out-of-memory errors.
    - Optimizing `per_device_train_batch_size` and `gradient_accumulation_steps`.
    - Detecting potential memory leaks during long training runs.
    """

    def __init__(self, log_every_n_steps: int = 10):
        """
        Initializes the callback.

        Args:
            log_every_n_steps (int, optional):
                The frequency of logging, in training steps. Defaults to 10.
        """
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event hook called by the Trainer at the end of each training step.

        This method checks if the current step is a logging step and, if so,
        prints the memory usage for all available CUDA devices.
        """
        # Log only every N steps to avoid cluttering the output.
        # The `is_world_process_zero` check ensures that this logging is only
        # performed by the main process in a multi-GPU (DDP) setup.
        if (
            state.global_step > 0
            and state.global_step % self.log_every_n_steps == 0
            and state.is_world_process_zero
        ):
            print(f"\n--- GPU Memory Stats at Step {state.global_step} ---")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                print(
                    f"GPU {i}: Allocated={allocated:.2f} MB | Reserved={reserved:.2f} MB"
                )
            print("------------------------------------")

def cache_vram(fraction: float = 0.95, device: int = 0):
    """
    Reserve approximately `fraction` of currently available GPU VRAM
    in PyTorch's CUDA caching allocator.
    """

    device = int(device)

    # Initialize CUDA context / allocator on this specific GPU.
    torch.empty(1, device=f"cuda:{device}")

    free, total = torch.cuda.mem_get_info(device)
    target = int(free * fraction)

    print(f"[cuda:{device}] GPU VRAM total: {total / 1024**3:.2f} GiB")
    print(f"[cuda:{device}] GPU VRAM free:  {free / 1024**3:.2f} GiB")
    print(f"[cuda:{device}] Caching:        {target / 1024**3:.2f} GiB")

    # Force PyTorch to acquire the VRAM.
    dummy = torch.empty(
        target,
        dtype=torch.uint8,
        device=f"cuda:{device}",
    )

    # Remove the live allocation. PyTorch's caching allocator retains
    # the underlying CUDA memory for subsequent allocations.
    del dummy

    reserved = torch.cuda.memory_reserved(device)

    print(
        f"[cuda:{device}] PyTorch reserved: "
        f"{reserved / 1024**3:.2f} GiB"
    )

def get_max_length(model: torch.nn.Module, config: PretrainedConfig = None) -> int:
    """
    Inspect model configuration to determine the maximum supported sequence length.

    Args:
        model (torch.nn.Module): Loaded base model.

    Returns:
        int: Maximum token length supported by the model configuration.
    """
    if model is None and config is None:
        raise Exception("Either model or config must be provided.")

    if model is not None:
        config = model.config

    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(config, length_setting, None)
        if max_length:
            break
    return max_length


def resolve_dataloader_drop_last(
    *,
    configured_drop_last: bool | None,
    dataset_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> bool:
    """Infer drop_last when exact elastic-resume geometry needs full batches."""

    if configured_drop_last is not None:
        return bool(configured_drop_last)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_global_batch = (
        world_size * per_device_train_batch_size * gradient_accumulation_steps
    )
    if effective_global_batch <= 0:
        raise ValueError("Effective global batch size must be positive.")

    remainder = dataset_length % effective_global_batch
    if remainder == 0:
        return False

    dropped = remainder
    print(
        "Inferred dataloader_drop_last=True because "
        f"train_dataset_size={dataset_length} is not divisible by "
        f"effective_global_batch_size={effective_global_batch}; "
        f"the last {dropped} example(s) will be skipped this epoch."
    )
    return True


def extract_last_json(text):
    """
    Extract the last valid JSON object or array from a text string.

    Args:
        text (str): The text containing JSON data

    Returns:
        dict/list: The parsed JSON object or array, or None if no valid JSON found
    """
    # Find all potential JSON objects {...} and arrays [...]
    # Using a regex to find balanced braces/brackets
    json_candidates = []

    # Find all positions where JSON might start
    for match in re.finditer(r"[{\[]", text):
        start = match.start()
        # Try to parse JSON from this position to the end
        for end in range(len(text), start, -1):
            substring = text[start:end]
            try:
                parsed = json.loads(substring)
                json_candidates.append((start, end, parsed))
                break  # Found valid JSON starting at this position
            except (json.JSONDecodeError, ValueError):
                continue

    # Return the last (rightmost) valid JSON found
    if json_candidates:
        # Sort by start position and return the last one
        json_candidates.sort(key=lambda x: x[0])
        return json_candidates[-1][2]

    return None
