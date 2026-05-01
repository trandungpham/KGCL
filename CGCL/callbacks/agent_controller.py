"""
AgentTrainingController
=======================
A PyTorch Lightning Callback that uses Claude (Anthropic API) as an agentic
controller during training.

Every `check_every_n_epochs` validation epochs the callback:
  1. Serialises the full metric history into a prompt.
  2. Calls Claude with a structured-output tool (`adjust_hyperparams`) that
     forces a typed JSON decision.
  3. Applies the returned adjustments directly to the live
     FinetuneModule / PretrainModule (lambda weights, LR scaling, early-stop).

Usage
-----
Add to your callbacks list in train_multitask.py:

    from callbacks.agent_controller import AgentTrainingController
    import anthropic

    agent_cb = AgentTrainingController(
        client=anthropic.Anthropic(),          # reads ANTHROPIC_API_KEY from env
        check_every_n_epochs=5,
        lambda_bounds=(0.1, 5.0),
        lr_scale_bounds=(0.5, 2.0),
    )
    callbacks.append(agent_cb)

Requirements
------------
    pip install anthropic
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema — Claude must respond with this structure
# ---------------------------------------------------------------------------
_ADJUST_TOOL = {
    "name": "adjust_hyperparams",
    "description": (
        "Return adjustments to apply to the running training job. "
        "Only include fields you actually want to change. "
        "Set stop_training=true to trigger early stopping."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lambda_diag": {
                "type": "number",
                "description": "New weight for the diagnosis loss (finetune phase only).",
            },
            "lambda_clue": {
                "type": "number",
                "description": "New weight for the clue-presence BCE loss.",
            },
            "lambda_chaos": {
                "type": "number",
                "description": "New weight for the chaos BCE loss.",
            },
            "lambda_align": {
                "type": "number",
                "description": "New weight for the clue-area spatial alignment loss.",
            },
            "lr_scale": {
                "type": "number",
                "description": (
                    "Multiplicative factor applied to *all* optimizer LR param groups "
                    "(e.g. 0.5 halves the LR, 2.0 doubles it)."
                ),
            },
            "stop_training": {
                "type": "boolean",
                "description": "If true, stop training immediately after this epoch.",
            },
            "reasoning": {
                "type": "string",
                "description": "One-sentence justification logged to the console.",
            },
        },
        "required": ["reasoning"],
    },
}

_SYSTEM_PROMPT = """\
You are an expert ML training controller for a skin-cancer multi-task model.

The model has four loss terms weighted by lambdas:
  - lambda_diag  : cross-entropy for MEL vs NV diagnosis (finetune only)
  - lambda_clue  : BCE for 9 dermoscopic clue-presence labels
  - lambda_chaos : BCE for 2 chaos indicators (structure / colour)
  - lambda_align : BCE for spatial clue-area mask alignment

Key metrics to watch:
  - val_diag_acc / val_diag_f1  : primary objective
  - val_clue_f1 / val_chaos_f1  : auxiliary tasks
  - train_loss_epoch             : overall training loss

Decision guidelines:
  1. If val_diag_acc is not improving for 3+ epochs, consider increasing
     lambda_diag and/or decreasing lambda_align (spatial alignment is expensive
     but may not always help diagnosis).
  2. If val_clue_f1 collapses while val_diag_acc improves, you may be over-
     weighting diagnosis — reduce lambda_diag slightly.
  3. If all losses plateau, try halving the LR (lr_scale=0.5).
  4. Do not make changes every interval — if training is healthy, return
     reasoning="No changes needed." and omit all adjustment fields.
  5. Keep lambdas in [0.1, 5.0] and lr_scale in [0.5, 2.0] per step.
  6. Only set stop_training=true if the model is clearly diverging or val
     metrics have not improved for 10+ epochs.
"""


def _format_history(history: list[dict[str, float]]) -> str:
    """Render metric history as a compact markdown table for the prompt."""
    if not history:
        return "No metrics yet."

    # Collect all keys that ever appeared
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in history:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    header = "| epoch | " + " | ".join(all_keys) + " |"
    sep = "|" + "---|" * (len(all_keys) + 1)
    rows = []
    for i, row in enumerate(history):
        vals = [f"{row.get(k, float('nan')):.4f}" for k in all_keys]
        rows.append(f"| {i + 1} | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


class AgentTrainingController(pl.Callback):
    """
    Calls Claude every `check_every_n_epochs` epochs to adaptively adjust
    loss weights and learning rate.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic()`` instance (must have ANTHROPIC_API_KEY set).
    check_every_n_epochs:
        How often (in validation epochs) to consult the agent.
    model_id:
        Claude model to use.
    lambda_bounds:
        (min, max) allowed range for any lambda after adjustment.
    lr_scale_bounds:
        (min, max) allowed multiplicative LR step per agent call.
    """

    def __init__(
        self,
        client: Any,
        check_every_n_epochs: int = 5,
        model_id: str = "claude-opus-4-6",
        lambda_bounds: tuple[float, float] = (0.1, 5.0),
        lr_scale_bounds: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__()
        self.client = client
        self.check_every_n_epochs = check_every_n_epochs
        self.model_id = model_id
        self.lambda_bounds = lambda_bounds
        self.lr_scale_bounds = lr_scale_bounds

        self._metric_history: list[dict[str, float]] = []
        self._epochs_since_check = 0

    # ------------------------------------------------------------------
    # PL hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Snapshot scalar metrics for this epoch
        snapshot = {
            k: v.item() if hasattr(v, "item") else float(v)
            for k, v in trainer.callback_metrics.items()
        }
        self._metric_history.append(snapshot)
        self._epochs_since_check += 1

        if self._epochs_since_check < self.check_every_n_epochs:
            return

        self._epochs_since_check = 0
        decision = self._consult_agent(trainer.current_epoch)
        if decision:
            self._apply_decision(decision, trainer, pl_module)

    # ------------------------------------------------------------------
    # Agent call
    # ------------------------------------------------------------------

    def _consult_agent(self, current_epoch: int) -> dict | None:
        history_table = _format_history(self._metric_history)
        user_message = (
            f"Epoch {current_epoch + 1} has just completed.\n\n"
            f"Metric history (one row per validation epoch):\n\n"
            f"{history_table}\n\n"
            "Analyse the trends and call `adjust_hyperparams` with any "
            "changes you recommend, or with only `reasoning` if no change is needed."
        )

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                tools=[_ADJUST_TOOL],
                tool_choice={"type": "required", "name": "adjust_hyperparams"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            logger.warning(f"[AgentController] API call failed: {exc}. Skipping adjustment.")
            return None

        # Extract the tool-use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "adjust_hyperparams":
                return block.input

        return None

    # ------------------------------------------------------------------
    # Apply decisions
    # ------------------------------------------------------------------

    def _apply_decision(
        self,
        decision: dict,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        reasoning = decision.get("reasoning", "")
        print(f"\n[AgentController] Epoch {trainer.current_epoch + 1}: {reasoning}")

        lo, hi = self.lambda_bounds

        for attr in ("lambda_diag", "lambda_clue", "lambda_chaos", "lambda_align"):
            if attr in decision:
                new_val = float(max(lo, min(hi, decision[attr])))
                if hasattr(pl_module, attr):
                    old_val = getattr(pl_module, attr)
                    setattr(pl_module, attr, new_val)
                    print(f"  {attr}: {old_val:.4f} → {new_val:.4f}")

        if "lr_scale" in decision:
            scale = float(
                max(self.lr_scale_bounds[0], min(self.lr_scale_bounds[1], decision["lr_scale"]))
            )
            optimizers = trainer.optimizers
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            for opt in optimizers:
                for pg in opt.param_groups:
                    old_lr = pg["lr"]
                    pg["lr"] = old_lr * scale
                    print(f"  LR: {old_lr:.2e} → {pg['lr']:.2e}")

        if decision.get("stop_training", False):
            print("[AgentController] Agent requested early stop.")
            trainer.should_stop = True
