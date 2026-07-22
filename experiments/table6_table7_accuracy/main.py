import os,utils,lm_eval,logging
import quant._get_args as get_args
import quant.ost_model_utils as ost_model_utils,utils.data_utils as data_utils,utils.eval_utils as eval_utils,utils.gptq_utils as gptq_utils
import torch,torch.nn as nn,torch.nn.functional as F,transformers,sys,datetime,datasets,lm_eval,lm_eval.tasks,accelerate
from loguru import logger
from quant.ost_train import rotate_smooth_train
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM,eval_logger
eval_logger.level = logging.ERROR

def main(args):
    transformers.set_seed(args.seed)
    lm = ost_model_utils.LM(args)
    lm.model.eval()
    test_loader = data_utils.get_loaders(args.eval_dataset,seed=args.seed,model=args.model,seqlen=lm.seqlen,eval_mode=True)  
    if args.pre_eval:
        pre_ppl = eval_utils.evaluator(lm.model,test_loader,utils.DEV,args,eval_samples=args.eval_samples)
        logger.info(f"Float ppl:{pre_ppl}")
        if args.lm_eval:
            eval_tasks(lm,args)
    if args.rotate:
        lm.fuse_layer_norms()
        lm.generate_rotate_parameters()
        
        if False: 
            lm.rotate_smooth_model_inplace()
            pre_ppl = eval_utils.evaluator(lm.model,test_loader,utils.DEV,args)
            logger.info(f"Fuse Layer Norms PPL:{pre_ppl}")
        if args.train_rotate:
            lm.set_quant_state(use_weight_quant=args.train_enable_wquant,use_act_quant=True,use_fully_quant=args.fully_quant)
            lm = rotate_smooth_train(args,lm)
        elif args.resume_path is not None:
            q = torch.load(args.resume_path)
            r=lm.model.load_state_dict(q, strict=False)
            logger.info(f"resume from {args.resume_path}")
        lm.rotate_smooth_model_inplace()
        if False: 
            dynamic_ppl = dynamic_eval(lm,test_loader,args,use_act_quant=True,use_fully_quant=args.fully_quant,use_weight_quant=args.train_enable_wquant)
            logger.info(f"After Dynamic ActQuant PPL:{dynamic_ppl}")
            if args.test_static:
                eval_tasks(lm,args)
                static_ppl = static_eval(lm,test_loader,args)
                logger.info(f"After Static ActQuant PPL:{static_ppl}")

    if args.w_bits < 16: 
        lm.set_quant_state(use_act_quant=False,use_fully_quant=False)  
        if args.force_clip:
            args.w_clip = True
        save_dict = {}
        if args.load_qmodel_path: 
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path, map_location="cpu", weights_only=False)
            lm.model.load_state_dict(save_dict["model"])
        elif args.w_gptq: 
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=lm.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(lm.model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: 
            quantizers = gptq_utils.rtn_fwrd(lm.model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        if args.save_qmodel_path:
            save_dict["model"] = lm.model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)
        lm.set_quant_state(use_act_quant=True,use_fully_quant=False)
    
    if not args.skip_ppl_eval:
        dynamic_ppl = dynamic_eval(lm,test_loader,args,use_act_quant=True,use_fully_quant=args.fully_quant)
        logger.info(f"After Dynamic Act&Weight Quant PPL:{dynamic_ppl}")
        if args.test_static:
            static_ppl = static_eval(lm,test_loader,args)
            logger.info(f"After Static Act&Weight Quant PPL:{static_ppl}")

    if not args.lm_eval:
        return
    else:
        eval_tasks(lm,args)


def eval_tasks(lm,args):
    utils.cleanup_memory()
    if args.distribute:
        utils.distribute_model(lm.model)
    else:
        lm.model.to(utils.DEV)
    task_manager = TaskManager()
    tasks = task_manager.match_tasks(args.tasks)
    lm_eval_batch_size = args.lm_eval_batch_size if args.lm_eval_batch_size > 0 else "auto"
    max_batch_size = 256 if lm_eval_batch_size == "auto" else args.lm_eval_batch_size
    hflm = HFLM(
        pretrained=lm.model,
        tokenizer=lm.tokenizer,
        batch_size=lm_eval_batch_size,
        max_batch_size=max_batch_size,
    )
    results = lm_eval.simple_evaluate(
        hflm,
        tasks=tasks,
        batch_size=lm_eval_batch_size,
        max_batch_size=max_batch_size,
        limit=_parse_lm_eval_limit(getattr(args, "lm_eval_limit", None)),
    )
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results['results'].items()}
    mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in results['results'].items()}
    mean_std_val =round(sum(std_vals.values()) / len(std_vals.values()), 4) 
    metric_vals['acc_avg'] = mean_acc_val
    results['results']['AVERAGE'] = {
        "acc,none":mean_acc_val,
        "acc_stderr,none":mean_std_val
    }
    logger.info("\n" + make_table(results))


def _parse_lm_eval_limit(raw_limit):
    if raw_limit in (None, "", "None", "none", "null", "full"):
        return None
    raw_text = str(raw_limit)
    if "." in raw_text:
        return float(raw_text)
    return int(raw_text)




def dynamic_eval(lm:ost_model_utils.LM,dataloader,args,use_act_quant=True,use_fully_quant=False,use_weight_quant=False):
    lm.set_quant_state(use_weight_quant=use_weight_quant,use_act_quant=use_act_quant,use_fully_quant=use_fully_quant)
    lm.set_dynamic(True)
    ppl = eval_utils.evaluator(lm.model,dataloader,utils.DEV,args,eval_samples=args.eval_samples)
    return ppl

def static_eval(lm:ost_model_utils.LM,dataloader,args):
    lm.set_quant_state(False,True,args.fully_quant)
    lm.set_dynamic(False)
    if not lm.calibrated:
        lm.set_observer(True)
        eval_utils.evaluator(lm.model,dataloader,utils.DEV,args,eval_samples=2) 
    lm.set_observer(False) 
    ppl = eval_utils.evaluator(lm.model,dataloader,utils.DEV,args,eval_samples=args.eval_samples)
    return ppl

if __name__ == "__main__":
    args = get_args.parse_args()
    main(args)
