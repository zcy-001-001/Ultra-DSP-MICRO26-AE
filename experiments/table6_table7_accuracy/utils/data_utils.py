import torch,datasets,random,transformers,os
from typing import Any, Dict
from datasets import Dataset
def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)

    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    model_type = model.split("/")[-1]
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_f = f'{cache_dir}/{name}_{model_type}_{"test" if eval_mode else "train"}_{nsamples}_{seqlen}_{seed}.cache'
    if os.path.exists(cache_f):
        # These caches store tokenized dataset objects, not bare tensor weights.
        loader = torch.load(cache_f, weights_only=False)
        print(f"load loader from {cache_f}")
    else:
        if 'wikitext2' in name:
            loader = get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
        if 'ptb' in name:
            loader = get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
        if 'c4' in name:
            loader = get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
        torch.save(loader, cache_f)
    return loader

class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        self.grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = self.grouped_dataset["input_ids"]
        self.labels = self.grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        
        
        concatenated_examples = {}

        
        for d in examples:
            
            for key in d.keys():
                
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        
        
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


def group_texts(block_size, examples):
    
    
    concatenated_examples = {}

    
    for d in examples:
        
        for key in d.keys():
            
            if key not in concatenated_examples:
                concatenated_examples[key] = []
            
            concatenated_examples[key].extend(d[key])
    total_length = len(concatenated_examples["input_ids"])
    
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    result = {
        k: [
            t[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
