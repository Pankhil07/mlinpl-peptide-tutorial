"""TASAR sampling: advantage-weighted incremental stochastic beam search.

Instead of sampling sequences independently from the language model, TASAR runs
several rounds of stochastic beam search. After each round the completed
sequences are scored by an oracle, and the log-probabilities of the tree nodes
that produced them are shifted by the resulting advantage. Later rounds
therefore spend their budget on the regions of sequence space the oracle likes,
while sampling without replacement keeps the beams from collapsing onto a
single candidate.

Vendored from the `joint-improvement` repository so that this repo stays
self-contained. The underlying stochastic beam search follows the official
TASAR implementation: https://github.com/grimmlab/graphxform

References
----------
Kool et al., "Stochastic Beams and Where to Find Them", ICML 2019.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import torch

from hyformer.generators.generator import GeneratorMixin
from hyformer.generators.utils.tasar.incremental_sbs import IncrementalSBS

if TYPE_CHECKING:
    from collections.abc import Callable

StateList = List[torch.Tensor]
StateTensor = torch.Tensor


class TasarMixin(GeneratorMixin):
    """TASAR (stochastic beam search) generation mixin.

    Expects `self` to behave like the language model: it must provide
    `_get_model_logits`, plus the `training` / `eval()` / `train()` surface of
    `torch.nn.Module`. Use :class:`TasarSampler` to wrap an existing Hyformer
    rather than mixing this into the model class, which would shadow the
    model's own `generate`.
    """

    def _cast_to_tensor(self, states: StateList) -> torch.Tensor:
        return torch.stack(states, dim=0)

    def _cast_to_states(self, tensor: torch.Tensor) -> StateList:
        return [tensor[batch_idx].squeeze(0) for batch_idx in range(tensor.shape[0])]

    def _build_fn(self, fn: "Callable[..., Any]", **kwargs: Any) -> "Callable[..., Any]":
        return functools.partial(fn, **kwargs)

    def _child_log_probability_fn(
        self, states: StateList, temperature: float = 1.0, top_k: Optional[int] = None
    ) -> List[np.ndarray]:
        logits = self._get_model_logits(self._cast_to_tensor(states))
        if temperature != 1.0:
            logits = self._scale_logits(logits, temperature=temperature)
        if top_k is not None:
            logits = self._apply_top_k(logits, top_k=top_k)
        log_probs = self._compute_log_probs(logits)
        return self._cast_to_states(log_probs.detach().cpu().numpy())

    def _child_transition_fn(
        self,
        state_action_pairs: List[Tuple[StateTensor, int]],
        max_sequence_length: int,
        eos_token_id: int,
    ) -> List[Tuple[StateTensor, bool]]:
        assert eos_token_id is not None, "EOS token ID must be provided."
        assert isinstance(eos_token_id, int), "EOS token ID must be an integer."
        new_states: List[Tuple[StateTensor, bool]] = []
        for prefix, action in state_action_pairs:
            new_ids = torch.cat(
                [prefix, torch.tensor([action], dtype=torch.long, device=prefix.device)], dim=0
            )
            is_leaf = (int(action) == int(eos_token_id)) or (new_ids.size(0) >= max_sequence_length)
            new_states.append((new_ids, bool(is_leaf)))
        return new_states

    def _leaf_evaluation_fn(
        self,
        input_ids: torch.LongTensor,
        advantage_fn: "Callable[[float], float]",
        oracle_fn: "Optional[Callable[[torch.LongTensor], float]]" = None,
    ) -> float:
        objective = (
            self._get_model_predictions(input_ids=input_ids) if oracle_fn is None else oracle_fn(input_ids)
        )
        return advantage_fn(float(objective))

    def _get_sampler(
        self,
        initial_states: StateList,
        child_log_probability_fn: "Callable[[StateList], List[np.ndarray]]",
        child_transition_fn: "Callable[..., List[Tuple[StateTensor, bool]]]",
        leaf_evaluation_fn: "Callable[[StateTensor], float]",
    ) -> IncrementalSBS:
        return IncrementalSBS(
            initial_states=initial_states,
            child_log_probability_fn=child_log_probability_fn,
            child_transition_fn=child_transition_fn,
            leaf_evaluation_fn=leaf_evaluation_fn,
            memory_aggressive=False,
        )

    def generate(
        self,
        prefix_input_ids: torch.LongTensor,
        advantage_fn: "Callable[[float], float]",
        eos_token_id: int,
        oracle_fn: "Optional[Callable[[StateTensor], float]]" = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        beam_width: int = 32,
        nucleus_top_p: float = 1.0,
        max_sequence_length: int = 128,
        deterministic: bool = False,
        replan_steps: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> List[torch.LongTensor]:
        """Generate sequences using incremental stochastic beam search.

        Parameters
        ----------
        prefix_input_ids : torch.LongTensor
            Prompt token IDs of shape (sequence_length,) for a single sequence.
            TASAR explores one tree at a time, so there is no batch dimension.
        advantage_fn : Callable[[float], float]
            Maps an oracle score to the advantage used to reweight the tree.
            Typically a constant rescaling, e.g. ``lambda r: 0.25 * r``.
        eos_token_id : int
            EOS token ID.
        oracle_fn : Callable[[torch.LongTensor], float], optional
            Scores a completed sequence (higher is better). Falls back to the
            model's own prediction head when None.
        temperature : float
            Temperature applied to the next-token logits.
        top_k : int, optional
            Top-k filtering on the next-token logits.
        beam_width : int
            Beam width for one round of SBS.
        nucleus_top_p : float
            Nucleus (top-p) threshold applied during beam expansion.
        max_sequence_length : int
            Maximum generated sequence length, prefix included.
        deterministic : bool
            Use deterministic beam search instead of stochastic sampling.
        replan_steps : int
            Number of SBS rounds; log-probs are updated after each round.
        rng : np.random.Generator, optional
            Generator for reproducible sampling. Uses the global NumPy random
            state when None.

        Returns
        -------
        List[torch.LongTensor]
            The sequences collected across all rounds, each of shape
            (sequence_length,).
        """
        was_training = self.training
        self.eval()

        try:
            child_log_probability_fn = self._build_fn(
                self._child_log_probability_fn, temperature=temperature, top_k=top_k
            )
            child_transition_fn = self._build_fn(
                self._child_transition_fn,
                max_sequence_length=max_sequence_length,
                eos_token_id=eos_token_id,
            )
            leaf_evaluation_fn = self._build_fn(
                self._leaf_evaluation_fn, advantage_fn=advantage_fn, oracle_fn=oracle_fn
            )

            sampler = self._get_sampler(
                # batch size == 1: TASAR explores a single tree
                initial_states=self._cast_to_states(prefix_input_ids.unsqueeze(0)),
                child_log_probability_fn=child_log_probability_fn,
                child_transition_fn=child_transition_fn,
                leaf_evaluation_fn=leaf_evaluation_fn,
            )

            result = sampler.perform_tasar(
                beam_width=beam_width,
                nucleus_top_p=nucleus_top_p,
                replan_steps=replan_steps,
                deterministic=deterministic,
                rng=rng,
            )
            return self._cast_tasar_result(result)
        finally:
            if was_training:
                self.train()

    def _cast_tasar_result(self, result: List[List[Any]]) -> List[torch.LongTensor]:
        return [leaf.state for leaf in result[0]]

    @torch.inference_mode()
    def _get_model_logits(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("Subclass must implement this method.")

    @torch.inference_mode()
    def _get_model_predictions(self, input_ids: torch.LongTensor) -> float:
        raise NotImplementedError("Subclass must implement this method.")


class TasarSampler(TasarMixin):
    """Wraps a Hyformer so it can be sampled with TASAR.

    The wrapper composes rather than patches: `model.generate` keeps its usual
    autoregressive meaning, and TASAR sampling is reached through
    `TasarSampler(model).generate(...)`.

    Parameters
    ----------
    model : torch.nn.Module
        A Hyformer whose `forward` accepts `task="lm"` / `task="prediction"`.
    prediction_task_type : str, optional
        `"regression"` or `"classification"`. Only needed when sampling without
        an explicit `oracle_fn`, so the prediction head is read correctly.
    """

    def __init__(self, model: torch.nn.Module, prediction_task_type: Optional[str] = None) -> None:
        self.model = model
        self.prediction_task_type = prediction_task_type or getattr(
            model, "prediction_task_type", None
        )

    # --- the torch.nn.Module surface TasarMixin.generate relies on ---
    @property
    def training(self) -> bool:
        return self.model.training

    def eval(self) -> "TasarSampler":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "TasarSampler":
        self.model.train(mode)
        return self

    @torch.inference_mode()
    def _get_model_logits(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            next_token_only=True,
            task="lm",
        )
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out

        # Normalize to (batch, 1, vocab): the SBS code expects exactly one
        # next-token distribution per beam.
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        elif logits.dim() == 3:
            if logits.size(1) != 1:
                logits = logits[:, -1:, :]
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        return logits

    @torch.inference_mode()
    def _get_model_predictions(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask, task="prediction", **kwargs
        )["logits"]
        if self.prediction_task_type == "classification":
            return torch.sigmoid(logits)
        if self.prediction_task_type == "regression":
            return logits
        raise ValueError(
            "`prediction_task_type` must be either `classification` or `regression`, "
            f"got {self.prediction_task_type!r}."
        )
