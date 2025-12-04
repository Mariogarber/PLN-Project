import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class CurriculumStage:
    """
    Configuration for a single curriculum stage.

    Attributes
    ----------
    stage : str
        Name of the curriculum bucket, e.g. "easy", "medium", "hard", "full".
    epochs : int
        Number of epochs to train on this stage.
    learning_rate : Optional[float]
        Optional learning rate override for this stage. If None, use base_args.learning_rate.
    max_steps : Optional[int]
        Optional maximum number of training steps. If None, use epoch-based training.
    resume_from_checkpoint : Optional[str]
        Optional checkpoint path to resume from at the beginning of this stage.
    """

    stage: str
    epochs: int
    learning_rate: Optional[float] = None
    max_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None


class CurriculumSeq2SeqTrainer:
    """
    High-level trainer that orchestrates multi-stage curriculum learning on top of Seq2SeqTrainer.

    It does NOT reimplement the training loop. Instead, it:
      - switches train_dataset across curriculum buckets (easy/medium/hard/full),
      - optionally adjusts LR/epochs per stage,
      - keeps the same model instance across stages,
      - logs metrics per stage.

    Typical usage:

        curriculum = dm.get_curriculum_splits()

        base_args = Seq2SeqTrainingArguments(
            output_dir="./mt5_detox_curriculum",
            ...
        )

        cur_trainer = CurriculumSeq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            curriculum_splits=curriculum,
            eval_dataset=val_dataset,
            base_training_args=base_args,
            compute_metrics=compute_metrics_fn,
            callbacks=[...],
            logger=logger,
        )

        schedule = [
            CurriculumStage(stage="easy",   epochs=2, learning_rate=2e-4),
            CurriculumStage(stage="medium", epochs=2, learning_rate=2e-4),
            CurriculumStage(stage="hard",   epochs=1, learning_rate=1e-4),
            CurriculumStage(stage="full",   epochs=1, learning_rate=1e-4),
        ]

        history = cur_trainer.run_schedule(schedule)

    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        data_collator: Callable,
        curriculum_splits: Dict[str, Dataset],
        eval_dataset: Optional[Dataset],
        base_training_args: Seq2SeqTrainingArguments,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.curriculum_splits = curriculum_splits
        self.eval_dataset = eval_dataset
        self.base_args = base_training_args
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.logger = logger

        self.history: List[Dict[str, Any]] = []

        # Sanity check
        if "easy" not in curriculum_splits and "full" not in curriculum_splits:
            raise ValueError(
                "Curriculum splits should contain at least 'full' or some of ['easy', 'medium', 'hard']."
            )

    # ------------------------------------------------------------------
    # Internal helper: build stage-specific TrainingArguments
    # ------------------------------------------------------------------
    def _make_stage_args(
        self,
        stage: str,
        epochs: int,
        learning_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Seq2SeqTrainingArguments:
        """
        Clone base args and adapt them for a given curriculum stage.
        """
        # Start from base args dict
        args_dict = self.base_args.to_dict()

        # Stage-specific output dir
        base_out = self.base_args.output_dir
        stage_out = os.path.join(base_out, f"stage_{stage}")
        args_dict["output_dir"] = stage_out

        # Override epochs and LR
        args_dict["num_train_epochs"] = epochs
        if learning_rate is not None:
            args_dict["learning_rate"] = learning_rate

        # Optional max_steps: if set, HF will prioritize steps over epochs
        if max_steps is not None:
            args_dict["max_steps"] = max_steps

        # Important: in transformers>=4.57, the argument is 'eval_strategy'
        # If your base_args already uses 'eval_strategy', this will be preserved.
        # We don't touch evaluation strategy here.

        stage_args = Seq2SeqTrainingArguments(**args_dict)
        return stage_args

    # ------------------------------------------------------------------
    # Train a single stage
    # ------------------------------------------------------------------
    def train_stage(self, stage_cfg: CurriculumStage) -> Dict[str, Any]:
        """
        Train the model on a single curriculum stage.

        Returns
        -------
        Dict[str, Any] : metrics dictionary from evaluation after this stage.
        """
        stage_name = stage_cfg.stage

        if stage_name not in self.curriculum_splits:
            raise ValueError(f"Stage '{stage_name}' not found in curriculum_splits.")

        train_dataset = self.curriculum_splits[stage_name]

        if self.logger:
            self.logger.info(
                f"[Curriculum] Starting stage '{stage_name}' with {len(train_dataset)} samples, "
                f"{stage_cfg.epochs} epochs, lr={stage_cfg.learning_rate}, "
                f"max_steps={stage_cfg.max_steps}"
            )

        # Build stage-specific training arguments
        stage_args = self._make_stage_args(
            stage=stage_name,
            epochs=stage_cfg.epochs,
            learning_rate=stage_cfg.learning_rate,
            max_steps=stage_cfg.max_steps,
        )

        # Create a new Seq2SeqTrainer for this stage, but share the same model
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=stage_args,
            train_dataset=train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train (optionally resuming)
        trainer.train(resume_from_checkpoint=stage_cfg.resume_from_checkpoint)

        # Evaluate after stage
        metrics = {}
        if self.eval_dataset is not None:
            metrics = trainer.evaluate()
            if self.logger:
                self.logger.info(f"[Curriculum] Metrics after stage '{stage_name}': {metrics}")

        # Keep updated model and record history
        self.model = trainer.model
        record = {
            "stage": stage_name,
            "output_dir": stage_args.output_dir,
            "metrics": metrics,
        }
        self.history.append(record)

        return record

    # ------------------------------------------------------------------
    # Train over a full schedule of stages
    # ------------------------------------------------------------------
    def run_schedule(self, schedule: List[CurriculumStage]) -> List[Dict[str, Any]]:
        """
        Run multiple curriculum stages sequentially.

        Parameters
        ----------
        schedule : List[CurriculumStage]
            Ordered list of stages (e.g. easy → medium → hard → full).

        Returns
        -------
        List[Dict[str, Any]]
            List of metrics records, one per stage.
        """
        if self.logger:
            self.logger.info(
                "[Curriculum] Starting curriculum schedule: "
                + " → ".join([s.stage for s in schedule])
            )

        for stage_cfg in schedule:
            self.train_stage(stage_cfg)

        if self.logger:
            self.logger.info("[Curriculum] Finished all stages.")

        return self.history

    # ------------------------------------------------------------------
    # Convenience: access to final model and history
    # ------------------------------------------------------------------
    def get_model(self) -> PreTrainedModel:
        """
        Return the underlying model (after last curriculum stage).
        """
        return self.model

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return the metrics history (one entry per stage).
        """
        return self.history
