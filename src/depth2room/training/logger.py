"""
Extended ModelLogger with Weights & Biases integration and checkpoint validation.

Subclasses the DiffSynth-Studio ModelLogger to add wandb experiment tracking,
validation loss computation, and distributed inference visualization at each
checkpoint save.
"""

import json
import logging
import math
import os
import traceback
import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffsynth.diffusion.logger import ModelLogger as _BaseModelLogger

log = logging.getLogger(__name__)


class ModelLogger(_BaseModelLogger):
    """ModelLogger with wandb integration, validation loss, and inference viz.

    Args:
        output_path: Directory for saving checkpoints.
        remove_prefix_in_ckpt: Prefix to strip from state dict keys.
        state_dict_converter: Optional function to transform the state dict.
        wandb_project: wandb project name. If None, wandb is disabled.
        wandb_entity: wandb entity (team/org). Defaults to "team-thomas".
        wandb_config: Dict of config to log to wandb.
        wandb_run_name: Optional run name for wandb.
        wandb_resume_id: If set, resume an existing wandb run with this ID.
        val_dataset: Optional dataset for validation at checkpoints.
        num_val_samples: Number of fixed samples for validation loss.
        val_seed: Random seed for selecting validation sample indices.
        model_dir: Path to base VACE model directory (enables inference viz).
        viz_inference_steps: Number of denoising steps for inference viz.
    """

    def __init__(self, output_path, remove_prefix_in_ckpt=None,
                 state_dict_converter=lambda x: x,
                 wandb_project=None, wandb_entity="team-thomas",
                 wandb_config=None, wandb_run_name=None,
                 wandb_resume_id=None,
                 val_dataset=None, num_val_samples=10, val_seed=42,
                 model_dir=None, viz_inference_steps=50):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.wandb_active = False
        self.wandb_run_id = None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.model_dir = model_dir
        self.viz_inference_steps = viz_inference_steps

        if wandb_project is not None:
            try:
                import wandb
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                if rank == 0:
                    if wandb_resume_id:
                        wandb.init(
                            project=wandb_project,
                            entity=wandb_entity,
                            id=wandb_resume_id,
                            resume="must",
                        )
                    else:
                        wandb.init(
                            project=wandb_project,
                            entity=wandb_entity,
                            config=wandb_config,
                            name=wandb_run_name,
                        )
                    self.wandb_active = True
                    self.wandb_run_id = wandb.run.id
            except ImportError:
                print("wandb not installed, skipping experiment tracking")

        # Validation setup: select fixed sample indices from the val dataset
        self.val_dataset = val_dataset
        if val_dataset is not None:
            import random as _rng
            rng = _rng.Random(val_seed)
            n = len(val_dataset)
            k = min(num_val_samples, n)
            self.val_indices = sorted(rng.sample(range(n), k))
            print(f"Validation enabled: {k} samples at indices {self.val_indices}")
        else:
            self.val_indices = []

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module,
                    save_steps=None, **kwargs):
        self.num_steps += 1

        if accelerator.is_main_process:
            loss = kwargs.get("loss")
            loss_val = None
            if loss is not None:
                loss_val = loss.item() if hasattr(loss, "item") else float(loss)
                if not math.isfinite(loss_val):
                    print(f"Step {self.num_steps}: NaN/Inf loss detected (loss={loss_val})")

            if self.wandb_active:
                import wandb
                log_dict = {"train/step": self.num_steps}
                if loss_val is not None:
                    log_dict["train/loss"] = loss_val
                lr = kwargs.get("learning_rate")
                if lr is not None:
                    log_dict["train/learning_rate"] = lr
                wandb.log(log_dict, step=self.num_steps)

        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if self.val_dataset is not None and self.val_indices:
                self._run_validation(accelerator, model)

    def _run_validation(self, accelerator, model):
        """Compute validation loss and run distributed inference visualization."""
        self._compute_val_loss(accelerator, model)
        if self.model_dir:
            self._run_distributed_inference_viz(accelerator, model)

    def _compute_val_loss(self, accelerator, model):
        """Compute validation loss on fixed samples, distributed across ranks.

        Each rank computes a different subset of validation samples (both with
        and without reference). Results are reduced across all ranks so rank 0
        can log the global average to wandb.
        """
        import torch.distributed as dist

        unwrapped = accelerator.unwrap_model(model)
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device

        # Save per-module training mode so we can restore exactly
        # (frozen models should stay in eval, only vace should be in train)
        module_training_state = {
            name: module.training
            for name, module in unwrapped.named_modules()
        }
        unwrapped.eval()

        # Build work items: each is (val_index, with_ref: bool)
        # N samples x 2 = 2N work items, distributed round-robin across ranks
        work_items = []
        for i in self.val_indices:
            work_items.append((i, True))   # with reference
            work_items.append((i, False))  # without reference
        my_items = [w for idx, w in enumerate(work_items) if idx % world_size == rank]

        local_loss_with_ref = 0.0
        local_loss_no_ref = 0.0
        local_count_with_ref = 0
        local_count_no_ref = 0

        with torch.no_grad():
            for val_idx, with_ref in my_items:
                try:
                    data = self.val_dataset[val_idx]
                    if not with_ref:
                        data = dict(data)
                        data["vace_reference_image"] = None
                    loss = unwrapped(data)
                    loss_val = loss.item()
                    if with_ref:
                        local_loss_with_ref += loss_val
                        local_count_with_ref += 1
                    else:
                        local_loss_no_ref += loss_val
                        local_count_no_ref += 1
                except Exception as e:
                    print(f"  [rank {rank}] Validation sample {val_idx} "
                          f"({'with' if with_ref else 'no'}_ref) failed: {e}")
                    traceback.print_exc()

        # Restore exact per-module training mode (not recursive .train())
        for name, module in unwrapped.named_modules():
            target = module_training_state.get(name, False)
            if module.training != target:
                module.training = target

        # All-reduce to sum losses and counts across ranks
        stats = torch.tensor(
            [local_loss_with_ref, local_count_with_ref,
             local_loss_no_ref, local_count_no_ref],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss_wr, total_count_wr, total_loss_nr, total_count_nr = stats.tolist()

        avg_with_ref = total_loss_wr / total_count_wr if total_count_wr > 0 else float("nan")
        avg_no_ref = total_loss_nr / total_count_nr if total_count_nr > 0 else float("nan")
        avg_combined = (avg_with_ref + avg_no_ref) / 2 if total_count_wr > 0 and total_count_nr > 0 else float("nan")

        if accelerator.is_main_process:
            print(f"  Step {self.num_steps}: val_loss={avg_combined:.6f} "
                  f"(with_ref={avg_with_ref:.6f}, no_ref={avg_no_ref:.6f}, "
                  f"{int(total_count_wr)}/{len(self.val_indices)} samples)")
            if self.wandb_active:
                import wandb
                wandb.log({
                    "val/loss": avg_combined,
                    "val/loss_with_ref": avg_with_ref,
                    "val/loss_no_ref": avg_no_ref,
                    "val/num_samples": int(total_count_wr),
                }, step=self.num_steps)

    def _run_distributed_inference_viz(self, accelerator, model):
        """Generate inference visualizations distributed across all ranks.

        Each rank generates at most one video using the in-memory training
        pipeline (zero extra GPU memory for model weights). Composes a
        side-by-side video (Generated | Depth | Validity) and saves to the
        shared filesystem. After a barrier, rank 0 logs all videos to wandb.
        """
        from depth2room.training.viz_worker import (
            frames_to_numpy, depth_to_turbo_frames,
            validity_to_frames, compose_sidebyside,
        )
        from diffsynth.utils.data import save_video

        import time

        unwrapped = accelerator.unwrap_model(model)
        pipe = unwrapped.pipe
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        step = self.num_steps

        viz_dir = os.path.join(self.output_path, f"viz_step-{step}")
        os.makedirs(viz_dir, exist_ok=True)

        # Save per-module training mode
        module_training_state = {
            name: module.training
            for name, module in unwrapped.named_modules()
        }
        # Save scheduler state (pipe.__call__ overrides it for inference)
        saved_scheduler = {}
        for attr in ("training", "sigmas", "timesteps"):
            val = getattr(pipe.scheduler, attr, None)
            saved_scheduler[attr] = val.clone() if isinstance(val, torch.Tensor) else val
        if hasattr(pipe.scheduler, "linear_timesteps_weights"):
            w = pipe.scheduler.linear_timesteps_weights
            saved_scheduler["linear_timesteps_weights"] = (
                w.clone() if isinstance(w, torch.Tensor) else w
            )

        try:
            unwrapped.eval()

            # N scenes x 2 (with/without ref) = 2N work items across all ranks
            work_items = []
            for i, val_idx in enumerate(self.val_indices):
                work_items.append((i, val_idx, True))
                work_items.append((i, val_idx, False))
            my_items = [w for idx, w in enumerate(work_items) if idx % world_size == rank]

            with torch.no_grad():
                for scene_idx, val_idx, with_ref in my_items:
                    ref_tag = "with_ref" if with_ref else "no_ref"
                    tag = f"s{scene_idx}_{ref_tag}"
                    print(f"  [rank {rank}] Generating viz {tag} (step {step})...")

                    try:
                        data = self.val_dataset[val_idx]
                        depth_tensor = data["vace_video_tensor"]
                        validity_mask = data.get("vace_validity_mask")
                        prompt = data["prompt"]

                        ref_image = None
                        if with_ref:
                            ref_val = data.get("vace_reference_image")
                            if ref_val is not None:
                                if isinstance(ref_val, list) and len(ref_val) > 0:
                                    ref_image = ref_val[0]
                                elif isinstance(ref_val, Image.Image):
                                    ref_image = ref_val

                        frames = pipe(
                            prompt=prompt,
                            negative_prompt="blurry, low quality, distorted",
                            vace_video=depth_tensor,
                            vace_validity_mask=validity_mask,
                            vace_reference_image=ref_image,
                            vace_scale=1.0,
                            seed=42,
                            height=self.val_dataset.height,
                            width=self.val_dataset.width,
                            num_frames=self.val_dataset.num_frames,
                            cfg_scale=5.0,
                            num_inference_steps=self.viz_inference_steps,
                            sigma_shift=5.0,
                        )

                        # Compose side-by-side: Generated | Depth | Validity
                        gen_np = frames_to_numpy(frames)
                        depth_np = depth_to_turbo_frames(depth_tensor)
                        if validity_mask is not None:
                            validity_np = validity_to_frames(validity_mask)
                        else:
                            validity_np = np.full_like(depth_np, 255)
                        sbs = compose_sidebyside(gen_np, depth_np, validity_np)

                        mp4_path = os.path.join(viz_dir, f"{tag}.mp4")
                        save_video([sbs[t] for t in range(sbs.shape[0])],
                                   mp4_path, fps=16, quality=5)

                        # Save metadata for rank 0 to read
                        with open(os.path.join(viz_dir, f"{tag}.json"), "w") as f:
                            json.dump({
                                "scene_idx": scene_idx, "val_idx": val_idx,
                                "with_ref": with_ref, "prompt": prompt[:200],
                            }, f)

                        # GT + reference saved once per scene (from with_ref run)
                        if with_ref:
                            gt_frames = data.get("video")
                            if gt_frames and len(gt_frames) > 0:
                                gt_np = frames_to_numpy(gt_frames)
                                sbs_gt = compose_sidebyside(gt_np, depth_np, validity_np)
                                save_video(
                                    [sbs_gt[t] for t in range(sbs_gt.shape[0])],
                                    os.path.join(viz_dir, f"s{scene_idx}_gt.mp4"),
                                    fps=16, quality=5,
                                )
                            if ref_image is not None:
                                ref_image.save(
                                    os.path.join(viz_dir, f"s{scene_idx}_ref.png"))

                        print(f"  [rank {rank}] Done: {tag}")
                    except Exception as e:
                        print(f"  [rank {rank}] Viz {tag} failed: {e}")
                        traceback.print_exc()

            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [rank {rank}] Inference viz failed: {e}")
            traceback.print_exc()
        finally:
            try:
                # Restore per-module training mode (must run even on failure)
                for name, module in unwrapped.named_modules():
                    target = module_training_state.get(name, False)
                    if module.training != target:
                        module.training = target

                # Restore scheduler state
                for attr, val in saved_scheduler.items():
                    setattr(pipe.scheduler, attr, val)
            except Exception as restore_err:
                print(f"  [rank {rank}] WARNING: state restore failed: {restore_err}")

            # All ranks MUST reach this barrier to avoid NCCL deadlock
            accelerator.wait_for_everyone()

        # Allow shared filesystem metadata to propagate across nodes
        time.sleep(2)

        if accelerator.is_main_process and self.wandb_active:
            self._log_viz_to_wandb(viz_dir, step)

    def _log_viz_to_wandb(self, viz_dir, step):
        """Collect mp4 files from all ranks and log to wandb."""
        import wandb

        log_dict = {}

        for fname in sorted(os.listdir(viz_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(viz_dir, fname)) as f:
                meta = json.load(f)

            scene_idx = meta["scene_idx"]
            ref_tag = "with_ref" if meta["with_ref"] else "no_ref"
            tag = f"s{scene_idx}"
            mp4_path = os.path.join(viz_dir, f"{tag}_{ref_tag}.mp4")
            if os.path.exists(mp4_path):
                log_dict[f"viz/{tag}_{ref_tag}"] = wandb.Video(
                    mp4_path, fps=16,
                    caption=f"step-{step} {tag} {ref_tag} | depth | validity",
                )

        for i in range(len(self.val_indices)):
            gt_path = os.path.join(viz_dir, f"s{i}_gt.mp4")
            if os.path.exists(gt_path):
                log_dict[f"viz/s{i}_gt"] = wandb.Video(
                    gt_path, fps=16,
                    caption=f"s{i} ground_truth | depth | validity",
                )
            ref_path = os.path.join(viz_dir, f"s{i}_ref.png")
            if os.path.exists(ref_path):
                log_dict[f"viz/s{i}_ref"] = wandb.Image(
                    ref_path, caption=f"s{i} reference",
                )

        if log_dict:
            wandb.log(log_dict, step=step)
            print(f"  Logged {len(log_dict)} viz items to wandb at step {step}")

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

        if self.val_dataset is not None and self.val_indices:
            self._run_validation(accelerator, model)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module,
                        save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            if self.val_dataset is not None and self.val_indices:
                self._run_validation(accelerator, model)

        if self.wandb_active and accelerator.is_main_process:
            import wandb
            wandb.finish()
