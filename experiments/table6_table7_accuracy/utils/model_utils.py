import torch
import transformers
import os
import logging
from typing import *
import models as local_llama_models
# Direct Hugging Face imports are kept as reference, but this checkout uses the
# local LLaMA implementation from models/modeling_llama.py for evaluation.
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2RMSNorm
except Exception as exc:
    logging.warning("Qwen2 classes are unavailable in this environment: %s", exc)
    Qwen2DecoderLayer = tuple()
    Qwen2ForCausalLM = tuple()
    Qwen2RMSNorm = tuple()

# Original eager OPT imports are left here for maintenance context:
# OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
# OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
# In the PICACHU LLaMA-only eval environment, the local transformers OPT module
# may depend on external nonlinear-approximation weights.  Do not let that
# optional OPT path break import of the LLaMA evaluator.
try:
    OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
    OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
except Exception as exc:
    logging.warning("OPT classes are unavailable in this environment: %s", exc)
    OPT_MODEL = tuple()
    OPT_LAYER = tuple()
# Original lazy-attribute access is kept for context:
# LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
# LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
# QWEN_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
# QWEN_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
# Transformers 4.51 does not expose these modeling modules as attributes until
# directly imported, so use the local explicit classes above.
LLAMA_MODEL = local_llama_models.LlamaForCausalLM
LLAMA_LAYER = local_llama_models.LlamaDecoderLayer
LLAMA_RMSNORM = local_llama_models.LlamaRMSNorm
QWEN_MODEL = Qwen2ForCausalLM
QWEN_LAYER = Qwen2DecoderLayer
QWEN_RMSNORM = Qwen2RMSNorm


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, QWEN_MODEL):
        return QWEN_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    elif isinstance(model, QWEN_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, QWEN_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_qwen(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype="auto",
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name.lower():
        return get_llama(model_name, hf_token)
    if 'qwen' in model_name.lower():
        return get_qwen(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    elif "qwen" in model_name:
        return get_qwen(model_name,hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN_MODEL):
        model_type = QWEN_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type) -> List[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == QWEN_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == QWEN_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == QWEN_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, LLAMA_RMSNORM)
    elif model_type == QWEN_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, QWEN_RMSNORM)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == QWEN_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  
                new_module = new_module_factory(module, int(name))
            else:  
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def get_layer_io_save_path(args,fig_prefix=False):
    return os.path.join(args.save_path, 'flayer_io', f'{args.layer_idx:03d}{fig_prefix}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type in (LLAMA_MODEL,QWEN_MODEL):
        captured_inputs = {
            "q_proj":[],
            'k_proj': [],  
            'v_proj': [],
            'o_proj': [],
            'gate_proj': [],  
            'up_proj': [],
            'down_proj': [],
        }

        captured_outputs = {
            "q_proj":[],
            'k_proj': [],  
            'v_proj': [],
            'o_proj': [],
            'gate_proj': [],  
            'up_proj':[],
            'down_proj': []
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    
    for seq in layer_input:
        layer(seq)
    
    
    
    
    

    
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
