import utils
import torch
import os
from tqdm import tqdm
from loguru import logger


@torch.no_grad()
def evaluator(model, testenc, dev, args, fig_prefix="",eval_samples=-1):

    utils.cleanup_memory(False)
    model.cpu()
    model.eval()

    model_name_lower = args.model.lower()
    # Original ordering checked "opt" first:
    # if 'opt' in args.model:
    # That misclassifies local paths such as <MODEL_DIR>/llama2/...
    # as OPT, so check LLaMA/Qwen identities before the optional OPT path.
    if 'meta' in args.model:
        llama_type = True
        opt_type = False
    elif 'llama' in model_name_lower or 'qwen' in model_name_lower:
        llama_type = True
        opt_type = False
    elif 'opt' in model_name_lower:
        opt_type = True
        llama_type = False
    else:
        raise ValueError(f'Unknown model {args.model}')


    use_cache = model.config.use_cache
    model.config.use_cache = False

    if opt_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    elif llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    
    input_ids = testenc.input_ids  
    nsamples = input_ids.numel() // model.seqlen  
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  

    batch_size = args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, eval_samples if eval_samples >= 1 else nsamples, batch_size)]
    nbatches = len(input_ids)

    
    
    dtype = model.lm_head.weight.dtype
    
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if llama_type:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
   
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if opt_type:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']
    if attention_mask is not None:
        attention_mask = attention_mask[:1]

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        
        
        
        
        
        
        

        for j in range(nbatches):
            if opt_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
            elif llama_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    
    

    if opt_type:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)

    elif llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    utils.cleanup_memory(False)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
    for i in tqdm(range(nbatches)):
        hidden_states = inps[i]
        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        hidden_states = hidden_states.to(dev) 
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        
        shift_labels = shift_labels.to(dev)
        shift_logits = shift_logits.to(dev)
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logger.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()

def dynamic_eval(lm,dataloader,args,use_act_quant=True,use_fully_quant=False,use_weight_quant=False):
    lm.set_quant_state(use_weight_quant=use_weight_quant,use_act_quant=use_act_quant,use_fully_quant=use_fully_quant)
    lm.set_dynamic(True)
    ppl = evaluator(lm.model,dataloader,args.dev,args)
    return ppl

def static_eval(lm,dataloader,args):
    lm.set_quant_state(False,True,args.fully_quant)
    lm.set_dynamic(False)
    if not lm.calibrated:
        lm.set_observer(True)
        evaluator(lm.model,dataloader,args.dev,args,eval_samples=2) 
    lm.set_observer(False) 
    ppl = evaluator(lm.model,dataloader,args.dev,args)
    return ppl
