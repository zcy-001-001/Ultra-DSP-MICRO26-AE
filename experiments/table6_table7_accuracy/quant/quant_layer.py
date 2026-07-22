import torch, torch.nn as nn, torch.nn.functional as F, math, warnings
from typing import *
from models.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaRMSNorm,
    LlamaAttention,
    Cache,
    repeat_kv,
    LlamaDecoderLayer,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from .quant_ops import *


def build_rotary_matrix(cos, sin):
    bsz, seq_len, head_dim = cos.size()
    cos2d = cos.reshape(-1, head_dim)
    sin2d = sin.reshape(-1, head_dim)
    weight = torch.diag_embed(cos2d)
    sin_diag = torch.diag_embed(sin2d)
    weight[:, head_dim // 2 :] -= sin_diag[:, : head_dim // 2]
    weight[:, : head_dim // 2] += sin_diag[:, head_dim // 2 :]
    return weight  


class QuantMLP(nn.Module):
    def __init__(self, org_module: nn.Module, config: LlamaConfig, args):
        super().__init__()

        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.weight_quant_params,
            args.norm_quant_params,
            args.gate_proj_quant_params,
            name="gate",
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.weight_quant_params,
            args.mul_quant_params,
            args.down_proj_quant_params,
            name="down",
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.weight_quant_params,
            args.norm_quant_params,
            args.up_proj_quant_params,
            name="up",
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.silu = QuantSiLU(
            args.silu_quant_params,
            nonlinear_int_method=getattr(args, "nonlinear_int_method", "disabled"),
            nonlinear_bits=getattr(args, "nonlinear_bits", 8),
            nonlinear_int_profile=getattr(args, "nonlinear_int_profile", "strict"),
        )
        self.mul = QuantMul(
            args.mul_quant_params,
            nonlinear_int_method=getattr(args, "nonlinear_int_method", "disabled"),
            nonlinear_bits=getattr(args, "nonlinear_bits", 8),
            nonlinear_int_profile=getattr(args, "nonlinear_int_profile", "strict"),
        )

    def forward(self, x, R_S_factors: Optional[dict] = dict()):
        up_proj_rst = self.up_proj(x, **R_S_factors)
        gate_proj_rst = self.gate_proj(x, **R_S_factors)
        act_rst = self.silu(gate_proj_rst, **R_S_factors)
        mul_rst = self.mul(up_proj_rst, act_rst)
        down_proj_rst = self.down_proj(mul_rst, **R_S_factors)
        return down_proj_rst


class QuantAttention(nn.Module):
    def __init__(self, org_module: LlamaAttention, config: LlamaConfig, args):
        super().__init__()

        self.layer_idx = org_module.layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.use_sdpa = args.use_sdpa

        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.norm_quant_params,
            args.q_proj_quant_params,
            name="q",
            attn_instance=self,
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.norm_quant_params,
            args.k_proj_quant_params,
            name="k",
            attn_instance=self,
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        # PICACHU Table 2 follow-up keeps RoPE on the original FP path by
        # default; nonlinear_quant_rope is only recorded as an experiment arg.
        self.ropeq = QuantROPE(act_quant_params=args.ropeq_quant_params)
        self.ropek = QuantROPE(act_quant_params=args.ropek_quant_params)
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.norm_quant_params,
            args.v_proj_quant_params,
            name="v",
            attn_instance=self,
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj,
            args.weight_quant_params,
            args.pv_matmul_quant_params,
            args.o_proj_quant_params,
            name="o",
            attn_instance=self,
            linear_int_mode=args.linear_int_mode,
            linear_int_accum_dtype=args.linear_int_accum_dtype,
            linear_int_variant=args.linear_int_variant,
            linear_int_fixed_alpha=args.linear_int_fixed_alpha,
        )
        self.qk_matmul = QuantMatmul(args.qk_matmul_quant_params, is_qkmat=True)
        self.pv_matmul = QuantMatmul(args.pv_matmul_quant_params, is_pvmat=True)

        self.softmax = QuantSoftmax(
            args.softmax_quant_params,
            -1,
            nonlinear_int_method=getattr(args, "nonlinear_int_method", "disabled"),
            nonlinear_bits=getattr(args, "nonlinear_bits", 8),
            nonlinear_int_profile=getattr(args, "nonlinear_int_profile", "strict"),
        )

        self.rotary_emb = org_module.rotary_emb

        self.pre_rope_Q = None
        self.post_rope_Q = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        R_S_factors: Optional[dict] = dict(),
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, **R_S_factors)
        key_states = self.k_proj(hidden_states, **R_S_factors)
        value_states = self.v_proj(hidden_states, **R_S_factors)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        )  
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)  
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )  
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        rotary_matrix = build_rotary_matrix(cos, sin).to(query_states.device)

        pre_rope_Q = self.pre_rope_Q.weight if self.pre_rope_Q is not None else None
        post_rope_Q = self.post_rope_Q.weight if self.post_rope_Q is not None else None
        key_states = self.ropek(key_states, rotary_matrix, pre_rope_Q, post_rope_Q).transpose(
            1, 2
        )  
        query_states = self.ropeq(query_states, rotary_matrix, pre_rope_Q, post_rope_Q).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_sdpa:
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            is_causal = True if causal_mask is None and q_len > 1 else False

            with torch.backends.cuda.sdp_kernel(True,True,True):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=causal_mask,
                    dropout_p=0,
                    
                    
                    is_causal=is_causal,
                )
            if self.pv_matmul.online_partial_had:
                init_shape = attn_output.shape  
                i_dtype = attn_output.dtype
                if self.pv_matmul.K == 1:  
                    
                    
                    if self.pv_matmul.fp32_had:
                        attn_output = (
                            fast_hadamard_transform.hadamard_transform(
                                attn_output.float().permute(0, 2, 3, 1),
                                scale=1 / math.sqrt(attn_output.shape[1]),
                            )
                            .permute(0, 3, 1, 2)
                            .to(i_dtype)
                        )
                    else:
                        attn_output = fast_hadamard_transform.hadamard_transform(
                            attn_output.permute(0, 2, 3, 1),
                            scale=1 / math.sqrt(attn_output.shape[1]),
                        ).permute(
                            0, 3, 1, 2
                        )  
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            if self.pv_matmul.use_act_quant:
                attn_output = self.pv_matmul.act_quantizer(attn_output)
            attn_output = self.o_proj(attn_output, **R_S_factors)
        else:
            attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3)) / (
                math.sqrt(self.head_dim)
            )  
            
            
            if attention_mask is not None:
                # Original eager path passed the cached mask through directly.
                # With current transformers masks can include one extra cache
                # column, so mirror the SDPA branch and slice to key length.
                attention_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = self.softmax(attn_weights, attention_mask).to(key_states.dtype)  
            attn_output = self.pv_matmul(attn_weights, value_states)  
            attn_output = self.o_proj(attn_output, **R_S_factors)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class QuantDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, ori_layer: LlamaDecoderLayer, args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantAttention(ori_layer.self_attn, config=config, args=args)
        self.mlp = QuantMLP(ori_layer.mlp, config=config, args=args)
        self.input_layernorm = QuantRMSNorm(
            ori_layer.input_layernorm,
            args.norm_quant_params,
            nonlinear_int_method=getattr(args, "nonlinear_int_method", "disabled"),
            nonlinear_bits=getattr(args, "nonlinear_bits", 8),
            nonlinear_int_profile=getattr(args, "nonlinear_int_profile", "strict"),
        )
        self.post_attention_layernorm = QuantRMSNorm(
            ori_layer.post_attention_layernorm,
            args.norm_quant_params,
            nonlinear_int_method=getattr(args, "nonlinear_int_method", "disabled"),
            nonlinear_bits=getattr(args, "nonlinear_bits", 8),
            nonlinear_int_profile=getattr(args, "nonlinear_int_profile", "strict"),
        )

        self.resadd1 = QuantAdd(args.resadd1_quant_params)
        self.resadd2 = QuantAdd(args.resadd2_quant_params)
        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_fully_quant = False
        self.R_S_modules = nn.ModuleDict(dict())
        self.temporary = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        R_res=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        
        
        R_S_factors = dict(R_res=R_res)
        R_S_factors.update(
            {key: value.weight for key, value in self.R_S_modules.items() if value is not None}
        )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states, S=R_S_factors.get("S_norm_qkv", None))

        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            R_S_factors=R_S_factors,
        )
        hidden_states = self.resadd1(residual, hidden_states)
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, S=R_S_factors.get("S_norm_upgate", None))
        hidden_states = self.mlp(hidden_states, R_S_factors=R_S_factors)
        hidden_states = self.resadd2(residual, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def set_quant_state(
        self,
        use_weight_quant: bool = False,
        use_act_quant: bool = False,
        use_fully_quant: bool = False,
    ):
        if use_fully_quant and (not use_act_quant):
            use_act_quant = True
            print("error: use_fully_quant must be used with use_act_quant")
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant
        self.use_fully_quant = use_fully_quant
        enable_linear_int = use_act_quant or use_fully_quant
        self.self_attn.q_proj.use_weight_quant = use_weight_quant
        self.self_attn.k_proj.use_weight_quant = use_weight_quant
        self.self_attn.v_proj.use_weight_quant = use_weight_quant
        self.self_attn.o_proj.use_weight_quant = use_weight_quant
        self.mlp.up_proj.use_weight_quant = use_weight_quant
        self.mlp.gate_proj.use_weight_quant = use_weight_quant
        self.mlp.down_proj.use_weight_quant = use_weight_quant
        self.self_attn.q_proj.enable_linear_int = enable_linear_int
        self.self_attn.k_proj.enable_linear_int = enable_linear_int
        self.self_attn.v_proj.enable_linear_int = enable_linear_int
        self.self_attn.o_proj.enable_linear_int = enable_linear_int
        self.mlp.up_proj.enable_linear_int = enable_linear_int
        self.mlp.gate_proj.enable_linear_int = enable_linear_int
        self.mlp.down_proj.enable_linear_int = enable_linear_int
        self.input_layernorm.use_act_quant = use_act_quant or use_fully_quant  
        self.self_attn.ropek.use_act_quant = use_act_quant or use_fully_quant  
        self.self_attn.v_proj.use_act_quant = use_act_quant or use_fully_quant  
        self.self_attn.pv_matmul.use_act_quant = use_act_quant or use_fully_quant  
        self.post_attention_layernorm.use_act_quant = use_act_quant or use_fully_quant  
        self.mlp.mul.use_act_quant = use_act_quant or use_fully_quant  
        self.self_attn.q_proj.use_act_quant = use_fully_quant
        self.self_attn.k_proj.use_act_quant = use_fully_quant
        self.self_attn.ropeq.use_act_quant = use_fully_quant
        self.self_attn.qk_matmul.use_act_quant = use_fully_quant
        self.self_attn.softmax.use_act_quant = use_fully_quant
        self.self_attn.o_proj.use_act_quant = use_fully_quant
        self.resadd1.use_act_quant = use_fully_quant
        self.mlp.up_proj.use_act_quant = use_fully_quant
        self.mlp.gate_proj.use_act_quant = use_fully_quant
        self.mlp.down_proj.use_act_quant = use_fully_quant
        self.mlp.silu.use_act_quant = use_fully_quant
        self.resadd2.use_act_quant = use_fully_quant

    def set_temporary(self, temporary: bool = False):
        self.temporary = temporary
        self.self_attn.q_proj.temporary = temporary
        self.self_attn.k_proj.temporary = temporary
        self.self_attn.v_proj.temporary = temporary
        self.self_attn.o_proj.temporary = temporary
        self.mlp.up_proj.temporary = temporary
        self.mlp.gate_proj.temporary = temporary
        self.mlp.down_proj.temporary = temporary
        self.mlp.silu.temporary = temporary

        self.input_layernorm.temporary = temporary
        self.post_attention_layernorm.temporary = temporary
