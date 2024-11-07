import time
import torch, os
from transformers import AutoTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, Dataset, Features, Value
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

accelerator = Accelerator()

modelpath = "state-spaces/mamba-1.4b"
bs = 4        
ga_steps = 1  
epochs = 25
lr = 0.00005
output_dir = "/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/retrain_checkpoints"

# Forward pass with custom loss
def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels=None):
    input_ids = input_ids.long()
    hidden_states = self.backbone(input_ids, inference_params=inference_params)
    if num_last_tokens > 0:
        hidden_states = hidden_states[:, -num_last_tokens:]
    lm_logits = self.lm_head(hidden_states)
    
    from torch.nn import CrossEntropyLoss
    if labels is not None:
        logits = lm_logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return (loss,)   
    else:
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

MambaLMHeadModel.forward = forward_with_loss

# Load model and tokenizer
model = MambaLMHeadModel.from_pretrained(
    modelpath,    
    dtype=torch.bfloat16,
    device="cuda",
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") 
tokenizer.pad_token = tokenizer.eos_token

# Resize tokenizer embeddings
def resize_token_embeddings(model, new_num_tokens):
    import torch.nn as nn

    old_embeddings = model.backbone.embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_embeddings = nn.Embedding(
        new_num_tokens,
        old_embedding_dim,
        device=old_embeddings.weight.device,
        dtype=old_embeddings.weight.dtype,
    )
    nn.init.normal_(new_embeddings.weight, std=0.02)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    model.backbone.embedding = new_embeddings
    model.tie_weights()

tokenizer.add_tokens(["<PAD>", "<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<|im_end|>"
resize_token_embeddings(model, len(tokenizer))

tokenizer.save_pretrained(f"{output_dir}/tokenizer/")

# Load and preprocess dataset
data_files = {"train": "/scratch/vetgpt/data/processed_data/redpajama_15_20_25_30_text/**/*.txt"} 
features = Features({"text": Value("string")})
dataset = load_dataset("text", data_files=data_files, features=features)

def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    
    remove_columns=["text"]
).filter(lambda x: len(x["input_ids"]) > 0)

def collate(elements):
    tokenlist = [e["input_ids"] for e in elements if len(e["input_ids"]) > 0] 
    tokens_maxlen = max([len(t) for t in tokenlist])

    input_ids, labels = [], []
    for tokens in tokenlist:
        pad_len = tokens_maxlen - len(tokens)
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch

# Define run name and set checkpoint timing
run_name = "{model}_{ds}_BS-{bs}_LR-{lr}".format(
    model=modelpath.split("/")[1],
    ds="processed_data",
    bs=bs,
    lr=lr,
)
run_name += "-ChatML"

# Calculate steps per 3-4 hours dynamically
steps_per_epoch = len(dataset_tokenized["train"]) // (accelerator.state.num_processes * bs * ga_steps)
approx_steps_per_hour = steps_per_epoch // 4
save_steps_interval = 3 * approx_steps_per_hour

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    evaluation_strategy="no",
    logging_steps=10,
    save_steps=save_steps_interval,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    learning_rate=lr,
    group_by_length=True,
    bf16=True,
    ddp_find_unused_parameters=False,
    save_safetensors=False,
    run_name=run_name,
)

# Assign configuration for the Trainer
model.config = model.config if hasattr(model.config, "to_dict") else model.config.__dict__

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
)

trainer.train()



# CUDA_VISIBLE_DEVICES=2,3,5,7 python /scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/models/finetune_mamba.py