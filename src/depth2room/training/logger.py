"""
Extended ModelLogger with Weights & Biases integration.

Subclasses the DiffSynth-Studio ModelLogger to add wandb experiment tracking
without modifying the upstream library. Logs training loss, step count,
epoch checkpoints, and training completion events.
"""

import os
import torch
from accelerate import Accelerator
from diffsynth.diffusion.logger import ModelLogger as _BaseModelLogger


class ModelLogger(_BaseModelLogger):
    """ModelLogger with wandb integration.

    Drop-in replacement for DiffSynth-Studio's ModelLogger. Adds optional
    wandb logging controlled by the wandb_project parameter.

    Args:
        output_path: Directory for saving checkpoints.
        remove_prefix_in_ckpt: Prefix to strip from state dict keys.
        state_dict_converter: Optional function to transform the state dict before saving.
        wandb_project: wandb project name. If None, wandb is disabled.
        wandb_config: Dict of config to log to wandb.
        wandb_run_name: Optional run name for wandb.
    """

    def __init__(self, output_path, remove_prefix_in_ckpt=None,
                 state_dict_converter=lambda x: x,
                 wandb_project=None, wandb_config=None, wandb_run_name=None):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.wandb_active = False

        if wandb_project is not None:
            try:
                import wandb
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                if rank == 0:
                    wandb.init(
                        project=wandb_project,
                        config=wandb_config,
                        name=wandb_run_name,
                    )
                    self.wandb_active = True
            except ImportError:
                print("wandb not installed, skipping experiment tracking")

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module,
                    save_steps=None, **kwargs):
        self.num_steps += 1

        if self.wandb_active and accelerator.is_main_process:
            import wandb
            log_dict = {"train/step": self.num_steps}
            loss = kwargs.get("loss")
            if loss is not None:
                log_dict["train/loss"] = loss.item() if hasattr(loss, "item") else float(loss)
            wandb.log(log_dict, step=self.num_steps)

        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)

            if self.wandb_active:
                import wandb
                wandb.log({"train/epoch": epoch_id, "train/checkpoint": path},
                          step=self.num_steps)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module,
                        save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

        if self.wandb_active and accelerator.is_main_process:
            import wandb
            wandb.finish()
