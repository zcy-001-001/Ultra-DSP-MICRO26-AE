import torch,torch.nn as nn,torch.nn.functional as F,matplotlib.pyplot as plt,numpy as np,os,re,tqdm
torch.set_grad_enabled(False)
from math import ceil

@torch.no_grad()
def get_and_plot(model,x,layer_idxs=[0,1,5,10,15,20,25,30,31],save_path=None,dev="cuda"):
    model = model.to(dev)
    x = x.to(dev)
    if save_path is None:
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
        save_path = f"fig_{now}"
    
    
    results = dict()
    def generate_hook(name,collect_input=True,collect_output=False,collect_weight=True):
        def hook(module,i,o):
            i = i[0]
            if collect_input:
                results[name+"_input"] = i.detach().cpu()
            if collect_output:
                results[name+"_output"] = o.detach().cpu()
            if collect_weight:
                results[name+"_weight"] = module.weight.detach().cpu()
        return hook
    layers = model.model.layers
    hooks = list()
    for i in layer_idxs:
        layer = layers[i]
        self_attn = layer.self_attn
        ffn = layer.mlp
        q,k,v,o,up,gate,down = (self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,ffn.gate_proj,ffn.up_proj,ffn.down_proj)
        for name,m in zip(("q","k","v","o","up","gate","down"),(q,k,v,o,up,gate,down)):
            m:nn.Module
            n = f"layer{i}_" + name
            hooks.append(m.register_forward_hook(generate_hook(n,collect_input=True,collect_output=name in ("k","v"),collect_weight=True)))
    model(x)
    for hook in hooks:
        hook.remove()
    
    
    from collections import defaultdict
    results_by_blocks = defaultdict(dict)
    for k,v in results.items():
        pattern = r"layer(\d+)"
        i = int(re.findall(pattern,k)[0])
        results_by_blocks[i][k] = v
    
    from tqdm import tqdm
    for idx,block in tqdm(results_by_blocks.items(),desc="plotting"):
        length = len(block)
        ncols = 1
        nrows = ceil(length/ncols)
        fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*8,nrows*4))
        axes = axes.reshape(-1)
        for ax,(k,v) in zip(axes,block.items()):
            if "weight" in k:
                value = v.T 
            else:
                value = v.flatten(0,-2) 
            value = value.float().cuda()
            pmax = torch.amax(value,dim=0).cpu().float().numpy()
            p9999 = torch.quantile(value,0.9999,dim=0).cpu().numpy()
            p99 = torch.quantile(value,0.99,dim=0).cpu().numpy()
            p75 = torch.quantile(value,0.75,dim=0).cpu().numpy()
            p25 = torch.quantile(value,0.25,dim=0).cpu().numpy()
            p01 = torch.quantile(value,0.01,dim=0).cpu().numpy()
            p0001 = torch.quantile(value,0.0001,dim=0).cpu().numpy()
            pmin = torch.amin(value,dim=0).cpu().numpy()
            x_label_ids = np.arange(len(pmin))
            ax.plot(x_label_ids,pmin,color='blue',label='Min/Max',linewidth=0.3)
            ax.plot(x_label_ids,p9999,color='red',label='1/9999 Percentile',linewidth=0.3)
            ax.plot(x_label_ids,p99,color='purple',label='1/99 Percentile',linewidth=0.3)
            ax.plot(x_label_ids,p75,color='orange',label='25/75 Percentile',linewidth=0.3)
            ax.plot(x_label_ids,p25,color='orange',linewidth=0.3)
            ax.plot(x_label_ids,p01,color='purple',linewidth=0.3)
            ax.plot(x_label_ids,p0001,color='red',linewidth=0.3)
            ax.plot(x_label_ids,pmax,color='blue',linewidth=0.3)
            ax.set_title(k)
            ax.set_xlabel("hidden dimension index")
            ax.set_ylabel("value")
            ax.legend(loc="upper right")
        os.makedirs(save_path,exist_ok=True)
        fig.tight_layout(rect=[0,0.05,1,0.95])
        fig.savefig(f'{save_path}/layer_{idx}',dpi=300)

    
    
    
    
    
    
def plot_simplify(model,test_loader,save_path=None,layer_idxs=[0,1,5,10,15,20,25,30,31],dev="cuda"):
    data = test_loader
    nsamples = data['input_ids'].numel() // 2048
    data = data['input_ids'].reshape(-1)[:nsamples*2048].reshape(-1,2048)
    x= data[0:100:10]
    get_and_plot(model,x,layer_idxs,save_path,dev)

def concat_plots(a_path, b_path, c_path):
    import glob
    from PIL import Image
    
    
    afs = [f.split("/")[-1] for f in glob.glob(a_path + "/*.png")]
    bfs = [f.split("/")[-1] for f in glob.glob(b_path + "/*.png")]
    common_names = sorted(list(set(afs).intersection(bfs)))
    if not len(common_names):
        return 
    os.makedirs(c_path,exist_ok=True)

    
    for name in tqdm.tqdm(common_names):
        img_a = Image.open(a_path + "/" + name)
        img_b = Image.open(b_path + "/" + name)
        
        
        width_a, height_a = img_a.size
        width_b, height_b = img_b.size
        
        total_width = width_a + width_b
        max_height = max(height_a, height_b)
        
        new_img = Image.new('RGB', (total_width, max_height))
        new_img.paste(img_a, (0, 0))
        new_img.paste(img_b, (width_a, 0))
        
        
        new_img.save(c_path + "/" + name)


    
if __name__ == "__main__"    :
    import random
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("weights/llama3-8b-hf",torch_dtype="auto",device_map="cuda")
    torch.set_grad_enabled(False)
    data = torch.load("cache/wikitext2_llama3-8b-hf_test_128_2048_0.cache")
    nsamples = data['input_ids'].numel() // 2048
    data = data['input_ids'].reshape(-1)[:nsamples*2048].reshape(-1,2048)
    model = model.cuda()
    x = data[0:100:10].cuda()
    get_and_plot(model,x)