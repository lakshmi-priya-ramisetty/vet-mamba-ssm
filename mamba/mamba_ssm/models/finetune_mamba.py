import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from trl import SFTTrainer
from peft import LoraConfig
from transformers import MambaConfig, MambaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

with open("/scratch/vetgpt/vetgpt-rlp/mamba/qa-pairs/train.json") as f:
    data = json.load(f)

def convert_to_hf_dataset(data):
    qa_pairs = [{"text": f"Question: {item['Question']} Answer: {item['Answer']}"} for item in data]
    return Dataset.from_list(qa_pairs)

hf_dataset = convert_to_hf_dataset(data)

train_test_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

checkpoint_path_model = '/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/retrain_checkpoints/checkpoint-517075'
config_path = "/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/checkpoints/checkpoint-3236/config.json"

config = MambaConfig.from_pretrained(config_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path_model)
model = MambaForCausalLM.from_pretrained(checkpoint_path_model, config=config, ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/mamba_finetune_new_qa_pairs_1.4b",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    evaluation_strategy="epoch", 
    learning_rate=2e-3,
    save_steps=100,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss", 
    overwrite_output_dir=True,
)

lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
)

trainer.train()

trainer.save_model("/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/mamba_finetune_new_qa_pairs_1.4b/final_model")
tokenizer.save_pretrained("/scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/mamba_finetune_new_qa_pairs_1.4b/final_model")

# CUDA_VISIBLE_DEVICES=3,5 python /scratch/vetgpt/vetgpt-rlp/mamba/mamba_ssm/models/finetune_new_qa_pairs_mamba.py
