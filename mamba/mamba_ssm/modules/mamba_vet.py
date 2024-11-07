import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import necessary functions and classes from the Mamba package
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
from mamba_ssm.mamba2_simple import Mamba2Simple

# Set up the device and tokenizer
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Data loading function
def load_data_from_directory(directory):
    data = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append(text)
    return data

# Define custom Dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data[idx]
        encoded = self.tokenizer.encode_plus(input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze()
        return input_ids

# Data loading and split
data_directory = '/scratch/vetgpt/data/cleaned_data/redpajama_15_20_25_30_text'
data = load_data_from_directory(data_directory)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(val_data, tokenizer)

# Collate function and data loaders
def collate_fn(batch):
    return torch.stack(batch, dim=0)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Veterinary Language Model with Mamba2Simple
class VeterinaryMambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_state=64, d_conv=4, chunk_size=256, dropout=0.1):
        super(VeterinaryMambaLM, self).__init__()
        
        # Embedding and Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))  # Assuming max length of 512

        # Mamba2Simple with SSD for structured state space processing
        self.mamba2_layer = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            chunk_size=chunk_size,
            activation="swish",
            use_mem_eff_path=True,
            device=device
        )
        
        self.norm = RMSNormGated(d_model, eps=1e-5, norm_before_gate=False, device=device)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Pass through Mamba2Simple layer
        mamba_output = self.mamba2_layer(embeddings)
        
        # Apply gating normalization and output projection
        normed_output = self.norm(mamba_output)
        logits = self.out_proj(normed_output)

        return logits

# Training and Validation Functions
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch.to(device)
            target_ids = batch.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss}")
        validate_model(model, val_loader, criterion)

def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch.to(device)
            target_ids = batch.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")

# Response Generation Function
def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    for _ in range(max_length):
        logits = model(generated)
        logits = logits[:, -1, :] / temperature  # Focus on the last token's logits
        probs = torch.softmax(logits, dim=-1)
        
        # Apply top-k sampling
        top_probs, top_indices = torch.topk(probs, top_k)
        next_token = top_indices[torch.multinomial(top_probs, 1)]
        
        generated = torch.cat((generated, next_token), dim=1)  # Append to sequence
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

# Model parameters and initialization
vocab_size = tokenizer.vocab_size
d_model = 512
d_state = 64
d_conv = 4
chunk_size = 256
epochs = 10

model = VeterinaryMambaLM(vocab_size, d_model, d_state, d_conv, chunk_size)
model.to(device)

# Start training
train_model(model, train_loader, val_loader, epochs)

# # Test with a prompt
# prompt = "What are the symptoms of heart disease in dogs?"
# response = generate_response(model, tokenizer, prompt)
# print("Bot response:", response)
