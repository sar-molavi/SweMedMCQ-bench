"""SFT trainer that evaluates EN and SV separately and exposes one aggregate loss."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

try:
    from trl import SFTTrainer
except ModuleNotFoundError:  # pragma: no cover - lets lightweight tests import helpers.
    class SFTTrainer:  # type: ignore[no-redef]
        pass


def dataset_weight(dataset: Any, weighting: str) -> float:
    if weighting == "equal":
        return 1.0
    if weighting != "num_examples":
        raise ValueError(f"Unsupported eval loss weighting: {weighting!r}")
    try:
        length = len(dataset)
    except TypeError as exc:
        raise ValueError("num_examples weighting requires eval datasets with __len__().") from exc
    if length <= 0:
        raise ValueError("Eval dataset weights must be positive.")
    return float(length)


def weighted_eval_loss(
    metrics_by_language: Mapping[str, Mapping[str, float]],
    eval_datasets: Mapping[str, Any],
    *,
    metric_key_prefix: str = "eval",
    weighting: str = "num_examples",
) -> float:
    """Compute the aggregate loss used for early stopping and best checkpointing."""

    numerator = 0.0
    denominator = 0.0
    for language, metrics in metrics_by_language.items():
        loss_key = f"{metric_key_prefix}_{language}_loss"
        if loss_key not in metrics:
            raise KeyError(f"Missing required metric {loss_key!r}.")
        weight = dataset_weight(eval_datasets[language], weighting)
        numerator += float(metrics[loss_key]) * weight
        denominator += weight
    if denominator <= 0:
        raise ValueError("Cannot aggregate eval loss with zero total weight.")
    return numerator / denominator


class _NoEvaluateCallbackProxy:
    """Delegate all callback handling except per-language on_evaluate events."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any) -> Any:
        return control


class BilingualEvalSFTTrainer(SFTTrainer):
    """Evaluate language-specific datasets and return an aggregate ``eval_loss``."""

    eval_loss_weighting: str = "num_examples"

    def __init__(self, *args: Any, eval_loss_weighting: str = "num_examples", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_loss_weighting = eval_loss_weighting

    @contextmanager
    def _suppress_sub_eval_callbacks(self):
        callback_handler = getattr(self, "callback_handler", None)
        if callback_handler is None:
            yield
            return
        self.callback_handler = _NoEvaluateCallbackProxy(callback_handler)
        try:
            yield
        finally:
            self.callback_handler = callback_handler

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        datasets = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(datasets, Mapping):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        if not datasets:
            raise ValueError("Bilingual evaluation requires at least one eval dataset.")

        all_metrics: dict[str, float] = {}
        metrics_by_language: dict[str, dict[str, float]] = {}
        with self._suppress_sub_eval_callbacks():
            for language, dataset in datasets.items():
                language_prefix = f"{metric_key_prefix}_{language}"
                metrics = super().evaluate(
                    eval_dataset=dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=language_prefix,
                )
                metrics_by_language[str(language)] = dict(metrics)
                all_metrics.update(metrics)

        aggregate_loss = weighted_eval_loss(
            metrics_by_language,
            {str(language): dataset for language, dataset in datasets.items()},
            metric_key_prefix=metric_key_prefix,
            weighting=self.eval_loss_weighting,
        )
        all_metrics[f"{metric_key_prefix}_loss"] = aggregate_loss

        self.log(all_metrics)
        callback_handler = getattr(self, "callback_handler", None)
        if callback_handler is not None:
            self.control = callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                all_metrics,
            )
        return all_metrics
