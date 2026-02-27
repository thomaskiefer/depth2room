"""
Training script for VACE depth-conditioned video generation.

Uses custom depth tensor loading, depth-aware VACE unit, and wandb-enabled
ModelLogger. Run via accelerate launch or the provided shell scripts.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
        -m depth2room.training.train \
        --task "sft" \
        --model_paths '[...]' \
        --dataset_base_path /path/to/data \
        --dataset_metadata_path /path/to/metadata.csv \
        --extra_inputs "vace_video_tensor,vace_reference_image" \
        --output_path /path/to/output
"""

import argparse
import logging
import os
import warnings

import accelerate
import torch

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    DirectDistillLoss,
    add_general_config,
    add_video_size_config,
    launch_training_task,
    launch_data_process_task,
)

from depth2room.training.dataset import VACEDepthDataset
from depth2room.training.training_unit import replace_vace_unit
from depth2room.training.logger import ModelLogger


class WanDepthTrainingModule(DiffusionTrainingModule):
    """
    Training module for depth-conditioned VACE training.

    Overrides:
      - __init__: replaces WanVideoUnit_VACE with WanVideoUnit_VACE_Depth
      - parse_extra_inputs: routes vace_video_tensor -> vace_video in inputs_shared
      - get_pipeline_inputs: handles video as list of PIL Images (uses PIL .size)
        and depth tensor correctly
    """

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. The training framework will forcibly enable it.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        # Replace the VACE unit with our depth-aware version BEFORE splitting
        self.pipe = replace_vace_unit(self.pipe)
        print("Replaced WanVideoUnit_VACE with WanVideoUnit_VACE_Depth")

        # Expand Context Embedder: 96 -> 160 input channels for validity mask
        # New 64 channels are zero-initialized so model starts from pretrained behavior
        old_conv = self.pipe.vace.vace_patch_embedding
        in_ch_old = old_conv.weight.shape[1]  # 96
        in_ch_new = in_ch_old + 64            # 160
        out_ch = old_conv.weight.shape[0]
        new_conv = torch.nn.Conv3d(
            in_ch_new, out_ch,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv = new_conv.to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        new_conv.weight.data.zero_()
        new_conv.weight.data[:, :in_ch_old] = old_conv.weight.data
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        self.pipe.vace.vace_patch_embedding = new_conv
        print(f"Expanded vace_patch_embedding: {in_ch_old} -> {in_ch_new} input channels")

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, None)

        # Training mode (full fine-tuning only, no LoRA)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            None, "", 32, None,
            None, None,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """
        Route extra inputs from the dataset dict into pipeline inputs.

        Key override: vace_video_tensor is routed to vace_video in inputs_shared
        so the depth-aware VACE unit receives it as a float tensor.
        """
        for extra_input in extra_inputs:
            if extra_input == "vace_video_tensor":
                # Route the pre-computed depth tensor to vace_video
                # The depth-aware unit will detect it's already a tensor and skip preprocess_video
                depth = data.get("vace_video_tensor")
                if depth is None:
                    logging.warning("vace_video_tensor is None — training on zero-depth fallback")
                inputs_shared["vace_video"] = depth
            elif extra_input == "vace_validity_mask":
                inputs_shared["vace_validity_mask"] = data.get("vace_validity_mask")
            elif extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                val = data.get(extra_input)
                if val is not None and isinstance(val, list) and len(val) > 0:
                    inputs_shared[extra_input] = val[0]
                else:
                    inputs_shared[extra_input] = val
            else:
                inputs_shared[extra_input] = data.get(extra_input)
        return inputs_shared

    def get_pipeline_inputs(self, data):
        """
        Build pipeline input dicts from dataset output.

        Key override: the base WanTrainingModule calls data["video"][0].size[1]
        which assumes PIL Images. We keep that for the video field (which IS
        a list of PIL Images from our dataset), but handle the depth tensor
        routing through parse_extra_inputs.
        """
        # video is a list of PIL Images from our dataset
        video_frames = data["video"]
        assert isinstance(video_frames, list) and len(video_frames) > 0, (
            f"Expected video to be a non-empty list of PIL Images, got {type(video_frames)}"
        )

        # PIL Image .size returns (width, height)
        frame_width, frame_height = video_frames[0].size

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video_frames,
            "height": frame_height,
            "width": frame_width,
            "num_frames": len(video_frames),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        if not torch.isfinite(loss):
            logging.error("Non-finite loss detected: %s", loss.item())
        return loss


def depth_train_parser():
    parser = argparse.ArgumentParser(description="VACE depth-conditioned training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to audio processor.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Initialize models on CPU.")
    parser.add_argument("--dry_run", default=False, action="store_true",
                        help="Load one sample, run one forward pass, print shapes/loss, then exit.")
    return parser


if __name__ == "__main__":
    parser = depth_train_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Use our custom depth dataset instead of UnifiedDataset
    dataset = VACEDepthDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        max_pixels=args.max_pixels,
        repeat=args.dataset_repeat,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Use our custom training module with depth-aware VACE unit
    model = WanDepthTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project="vace-depth-finetune",
        wandb_config=vars(args),
        wandb_run_name=f"full-lr{args.learning_rate}",
    )
    if args.dry_run:
        print("\n=== DRY RUN: loading one sample and running one forward pass ===")
        data = dataset[0]
        print(f"  prompt: {data['prompt'][:80]}...")
        print(f"  video: {len(data['video'])} frames, size={data['video'][0].size}")
        depth = data.get("vace_video_tensor")
        if depth is not None:
            print(f"  depth: shape={depth.shape}, dtype={depth.dtype}, "
                  f"range=[{depth.min():.3f}, {depth.max():.3f}]")
        else:
            print("  depth: None")
        validity = data.get("vace_validity_mask")
        if validity is not None:
            print(f"  validity mask: shape={validity.shape}, "
                  f"valid_frac={validity.mean():.3f}")
        else:
            print("  validity mask: None")
        ref = data.get("vace_reference_image")
        print(f"  reference: {'yes' if ref else 'no'}")

        # Save debug artifacts
        debug_dir = os.path.join(args.output_path, "dry_run")
        dataset.save_debug_sample(0, debug_dir)
        print(f"  Debug artifacts saved to {debug_dir}")

        # Forward pass
        model.to(accelerator.device)
        loss = model(data)
        print(f"  loss: {loss.item():.6f} (finite={torch.isfinite(loss).item()})")
        print("=== DRY RUN COMPLETE ===")
    else:
        launcher_map = {
            "sft:data_process": launch_data_process_task,
            "direct_distill:data_process": launch_data_process_task,
            "sft": launch_training_task,
            "sft:train": launch_training_task,
            "direct_distill": launch_training_task,
            "direct_distill:train": launch_training_task,
        }
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
