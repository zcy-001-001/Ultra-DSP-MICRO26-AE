import torch, torch.nn as nn, torch.nn.functional as F, argparse, transformers, sys, lm_eval, pprint
from transformers import Seq2SeqTrainingArguments
from typing import Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

def parse_args():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    init_logger(args)
    if (args.online_hadamard == "all" or args.online_hadamard == "v") and args.rotate_ov:
        logger.warning("on_line_hadamard and rotate_ov are both enabled, we will disalbe rotate_ov")
        args.rotate_ov = False
    if args.rotate_post_rope and args.online_qk_hadamard:
        logger.warning("online_qk_hadamard and rotate_post_rope are both enabled, we will disalbe online_qk_hdamard")
        args.online_qk_hadamard = False
    if args.nonlinear_quant_rope:
        logger.warning(
            "nonlinear_quant_rope=True was requested, but this PICACHU follow-up "
            "keeps QuantROPE on the original FP path."
        )

    args.embed_quant_params = dict(bits=args.residual_bits, sym=not args.a_asym, dynamic=True, dynamic_method="perchannel")
    
    args.weight_quant_params = dict(
        bits=args.w_bits,
        sym=not args.w_asym,
        groupsize=args.w_groupsize,
        dynamic=True,
        dynamic_method="pertoken",
        mse=args.w_clip,
    )
    
    args.norm_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.ropek_quant_params = dict(
        bits=args.k_bits,
        sym=not args.k_asym,
        groupsize=args.k_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.v_proj_quant_params = dict(
        bits=args.v_bits,
        sym=not args.v_asym,
        groupsize=args.v_groupsize,
        clip_ratio=args.v_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.pv_matmul_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=128,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.mul_quant_params = dict(
        bits=args.down_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    
    
    args.q_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.ropeq_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.k_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.k_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.qk_matmul_quant_params = dict(
        bits=args.attn_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.softmax_quant_params = dict(
        bits=args.attn_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.o_proj_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    args.resadd1_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    args.up_proj_quant_params = dict(
        bits=args.a_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.gate_proj_quant_params = dict(
        bits=args.act_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.silu_quant_params = dict(
        bits=args.act_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method=args.a_dynamic_method,
    )  
    args.down_proj_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )  
    args.resadd2_quant_params = dict(
        bits=args.residual_bits,
        sym=not args.a_asym,
        groupsize=args.a_groupsize,
        clip_ratio=args.k_clip_ratio,
        dynamic=True,
        dynamic_method="perchannel",
    )
    if args.narrow_symmetric:
        for name, value in vars(args).items():
            if name.endswith("_quant_params") and isinstance(value, dict):
                value["narrow_symmetric"] = True
    logger.debug(f"Arguments:{pprint.pformat(vars(args))}")
    logger.debug("--" * 30)
    return args


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    
    model: str = field(default="weights/llama3-8b-hf", metadata=dict(help="Path to the model"))
    fully_quant: bool = field(default=False, metadata=dict(help="Whether to use fully quantization"))
    test_static: bool = field(default=False, metadata=dict(help="Whether to use static quantization for activation"))
    use_sdpa: bool = field(default=True, metadata=dict(help="Use SDPA instead of default MHA"))

    train_distribute:bool = field(default=False, metadata=dict(help="Whether to use distributed training"))
    train_rotate: bool = field(default=True, metadata=dict(help="Whether to train with rotation"))
    max_steps: int = field(default=200, metadata=dict(help="Maximum number of training steps"))
    loss_type: str = field(
        default="origin",
        metadata=dict(
            choices=[
                "origin",
                "mse",
                "kl",
                "kd",
                "feature_mse",
                "r_kl_top",
                "rkl",
                "kl_top",
                "kl_top_5",
                "kl_top_10",
                "kl_top_50",
                "kl_top_100",
                "kl_top_500",
            ],
            help="Loss type for training",
        ),
    )
    opt_type: str = field(default="RAdam", metadata=dict(choices=["SGDG", "RAdam", "RSGD"], help="Optimizer type for training"))
    rotate_lr: float = field(default=0.01689753172873217, metadata=dict(help="Learning rate for rotation"))
    smooth_lr: float = field(default=0.0017898822675917248, metadata=dict(help="Learning rate for smoothing"))
    rotate_momentom: float = field(default=0, metadata=dict(help="Momentum for rotation"))
    smooth_momentom: float = field(default=0.9, metadata=dict(help="Momentum for smoothing"))
    train_enable_wquant: bool = field(default=False, metadata=dict(help="Enable weight quantization during training"))
    train_dataset: str = field(default="wikitext2", metadata=dict(choices=["wikitext2", "c4", "pdb"], help="Dataset for training"))
    resume_path: str = field(default=None, metadata=dict(help="Path to resume training"))
    
    per_device_train_batch_size: int = field(default=8, metadata=dict(help="Batch size per device for training"))
    per_device_eval_batch_size: int = field(default=8, metadata=dict(help="Batch size per device for evaluation"))
    
    
    rotate: bool = field(default=True, metadata=dict(help="Whether to rotate the model"))
    rotate_mode: str = field(default="hadamard", metadata=dict(choices=["hadamard", "random"], help="Initial rotation mode"))
    rotate_ov: bool = field(default=False, metadata=dict(help="Rotate V's output and O's input"))
    rotate_pre_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's input"))
    rotate_post_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's output"))
    rotate_rope_perlayer: bool = field(default=True, metadata=dict(help="Whether to allow each layer to have a separate ROPE matrix"))
    smooth_up_down: bool = field(default=False, metadata=dict(help="Smooth Up's output and Down's output"))
    smooth_up_gate: bool = field(default=False, metadata=dict(help="Smooth x1 and x2"))
    smooth_qk: bool = field(default=False, metadata=dict(help="Smooth q and k"))
    smooth_ov: bool = field(default=False, metadata=dict(help="Smooth v and o"))
    smooth_norm_linear: bool = field(default=False, metadata=dict(help="Smooth norm and linear after rotation"))

    rotate_down_dim: int = field(default=1, metadata=dict(help="Dimension for rotating down projection"))
    fp32_had: bool = field(default=True, metadata=dict(help="Apply Hadamard rotation in FP32"))
    online_hadamard: str = field(
        default="down",
        metadata=dict(choices=["all", "v", "down", "None"], help="Online Hadamard transformation settings"),
    )
    online_qk_hadamard: bool = field(default=True, metadata=dict(help="Apply online Hadamard to Q/K"))
    
    a_bits: int = field(default=4, metadata=dict(help="Number of bits for inputs of linear layers"))
    a_dynamic_method: str = field(
        default="pertoken",
        metadata=dict(choices=["pertoken", "perchannel", "pertensor"], help="Dynamic quantization method"),
    )
    a_groupsize: int = field(default=-1, metadata=dict(help="Groupsize for activation quantization"))
    a_asym: bool = field(default=True, metadata=dict(help="Asymmetric activation quantization"))
    a_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for activation quantization"))
    a_static: bool = field(default=False, metadata=dict(help="Whether to use static quantization for activation"))
    
    w_bits: int = field(default=4, metadata=dict(help="Number of bits for weights of linear layers"))
    w_groupsize: int = field(default=-1, metadata=dict(help="Groupsize for weight quantization"))
    w_asym: bool = field(default=False, metadata=dict(help="Asymmetric weight quantization"))
    narrow_symmetric: bool = field(default=False, metadata=dict(help="Use narrow symmetric range for symmetric quantizers"))
    w_clip: bool = field(default=True, metadata=dict(help="Clip weights during quantization"))
    force_clip: bool = field(default=True, metadata=dict(help="Clip weights during GPTQ"))
    w_gptq: bool = field(default=True, metadata=dict(help="Use GPTQ for weight quantization"))
    linear_int_mode: str = field(
        default="disabled",
        metadata=dict(choices=["disabled", "exact", "approx"], help="Replace quantized linear F.linear with integer exact/approx GEMM"),
    )
    linear_int_accum_dtype: str = field(
        default="bf16",
        metadata=dict(choices=["fp32", "bf16", "fp16"], help="Input dtype used by the accelerated approx integer GEMM path"),
    )
    linear_int_variant: str = field(
        default="lsb3_zero",
        metadata=dict(
            choices=["lsb3_zero", "fixed_alpha", "wp521_unsigned_raw"],
            help="Approximate multiplier variant used for Table 6 regeneration",
        ),
    )
    linear_int_fixed_alpha: float = field(
        default=1.0,
        metadata=dict(help="Per-product constant offset alpha used when linear_int_variant=fixed_alpha"),
    )
    nonlinear_int_method: str = field(
        default="disabled",
        metadata=dict(
            choices=["disabled", "ibert", "gemmlowp"],
            help="Emulate INT8 nonlinear operators for the PICACHU Table 2 follow-up",
        ),
    )
    nonlinear_bits: int = field(default=8, metadata=dict(help="Bit width for nonlinear fake quantization"))
    nonlinear_int_profile: str = field(
        default="strict",
        metadata=dict(
            choices=[
                "strict",
                "w8a8",
                "w8a8_mid",
                "w8a8_lossy",
                "fixedpoint",
                "softmax_int8",
                "no_mul",
                "softmax_only",
                "rmsnorm_only",
                "silu_only",
                "mul_only",
                "no_rmsnorm",
                "softmax_silu",
                "softmax_silu_mul",
                "softmax_fp",
                "rmsnorm_noinput_only",
                "noinput_rmsnorm",
                "noinput_rmsnorm_out8",
            ],
            help=(
                "Gemmlowp emulation profile. strict preserves the initial all-INT8-output "
                "fake-quant path; w8a8 preserves the rejected all-tensor/no-renorm "
                "diagnostic; w8a8_mid uses per-token activation scales plus INT8 "
                "nonlinear outputs and softmax renorm; w8a8_lossy keeps per-token "
                "scales but leaves INT8 softmax probability mass un-repaired; "
                "fixedpoint keeps gemmlowp-style fixed-point "
                "nonlinear outputs in FP/BF16 for comparison against PICACHU Table 2."
            ),
        ),
    )
    nonlinear_quant_rope: bool = field(
        default=False,
        metadata=dict(help="Record whether RoPE should be quantized; kept False for the PICACHU follow-up"),
    )
    nsamples: int = field(default=128, metadata=dict(help="Number of calibration data samples for GPTQ"))
    cal_dataset: str = field(
        default="wikitext2",
        metadata=dict(choices=["wikitext2", "pdb", "c4"], help="Calibration dataset for GPTQ"),
    )
    percdamp: float = field(default=0.01, metadata=dict(help="Percent of the average Hessian diagonal to use for dampening"))
    act_order: bool = field(default=False, metadata=dict(help="Activation order in GPTQ"))

    
    v_bits: int = field(default=4, metadata=dict(help="Number of bits for V-cache quantization"))
    v_groupsize: int = field(default=128, metadata=dict(help="Groupsize for V-cache quantization"))
    v_asym: bool = field(default=True, metadata=dict(help="Asymmetric V-cache quantization"))
    v_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for V-cache quantization"))
    k_bits: int = field(default=4, metadata=dict(help="Number of bits for K-cache quantization"))
    k_groupsize: int = field(default=128, metadata=dict(help="Groupsize for K-cache quantization"))
    k_asym: bool = field(default=True, metadata=dict(help="Asymmetric K-cache quantization"))
    k_clip_ratio: float = field(default=1.0, metadata=dict(help="Clip ratio for K-cache quantization"))
    
    residual_bits: int = field(default=16, metadata=dict(help="Bits for residual inputs and outputs"))
    attn_bits: int = field(default=16, metadata=dict(help="Bits for attention outputs"))
    act_bits: int = field(default=16, metadata=dict(help="Bits for activation"))
    down_bits: int = field(default=4, metadata=dict(help="Bits for down projection input"))
    int8_down_proj:bool = field(default=False, metadata=dict(help="Whether to use int8 for down projection"))

    
    load_qmodel_path: Optional[str] = field(default=None, metadata=dict(help="Path to load the quantized model"))
    save_qmodel_path: Optional[str] = field(default=None, metadata=dict(help="Path to save the quantized model"))

    
    eval_dataset: str = field(default="wikitext2", metadata=dict(choices=["wikitext2", "pdb", "c4"], help="Dataset for evaluation"))
    bsz: int = field(default=4, metadata=dict(help="Batch size for evaluation"))
    eval_samples: int = field(default=-1, metadata=dict(help="Number of evaluation chunks to use, -1 means full dataset"))
    skip_ppl_eval: bool = field(default=False, metadata=dict(help="Skip post-quantization PPL evaluation; useful when only saving regenerated qmodels"))
    lm_eval: bool = field(default=False, metadata=dict(help="Evaluate the model on LM Eval tasks"))
    tasks: List[str] = field(
        default_factory=lambda: [
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "lambada_openai",
            "openbookqa",
            "piqa",
            "social_iqa",
            "winogrande",
        ],
        metadata=dict(help="List of LM Eval tasks"),
    )
    lm_eval_batch_size: int = field(default=16, metadata=dict(help="Batch size for evaluating with LM Eval harness"))
    lm_eval_limit: Optional[str] = field(
        default=None,
        metadata=dict(help="Optional lm-eval limit; use an integer for smoke tests or leave unset for full eval"),
    )
    distribute: bool = field(default=False, metadata=dict(help="Distribute the model on multiple GPUs for evaluation"))

    pre_eval: bool = field(default=False, metadata=dict(help="Whether to evaluate the model before training"))
    
    qwen2_downfill: bool = field(default=False, metadata=dict(help="Whether to fill the down projection with zeros"))
    gradient_checkpointing: bool = field(default=True, metadata=dict(help="Whether to use gradient checkpointing"))
    logging_steps: int = field(default=1, metadata=dict(help="Number of steps between logging"))
    log_on_each_node: bool = field(default=False, metadata=dict(help="Whether to log on each node"))
    fp16: bool = field(default=False, metadata=dict(help="Whether to use fp16"))
    eval_strategy: str = field(default="steps", metadata=dict(choices=["steps", "epoch"], help="Evaluation strategy"))
    eval_steps: float = field(default=0.5, metadata=dict(help="Number of steps between evaluation"))
    save_strategy: str = field(default="no", metadata=dict(choices=["steps", "epoch", "no"], help="Save strategy"))
    fsdp_config: Optional[Union[dict, str]] = field(
        default_factory=lambda: {
            "cpu_ram_efficient_loading": True,
            
            "transformer_layer_cls_to_wrap": ["QuantDecoderLayer","QuantLinear"]
        },
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    report_to:str = field(default="none", metadata=dict(choices=["none", "wandb", "tensorboard"], help="Where to report metrics"))
    force_rdtype_inplace: bool = field(default=False,metadata=dict(help="when inplace weather use rtype if not we use fp64 to merge weights"))
    use_klt: bool = field(default=False,metadata=dict(help="whether to use klt"))


    sub_mean:bool = field(default=True,metadata=dict(help="whether to use sub mean"))
    post_attn: bool = field(default=False,metadata=dict(help="whether to use post attn for calculate kl loss"))

def init_logger(args):
    logger.remove()
    if args.local_rank in (0, -1):
        logger.add(sys.stdout, level="INFO")
        output_dir = args.output_dir
        log_file = output_dir + "/log.txt"
        logger.add(log_file, level="DEBUG")


if __name__ == "__main__":
    args = parse_args()
    print(args)
