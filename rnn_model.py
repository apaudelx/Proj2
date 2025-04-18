import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(model, tokenizer, test_samples, max_seq_length=100, device='cuda'):
    model.eval()
    bleu_scores = []
    smooth = SmoothingFunction().method1  
    
    for sample in tqdm(test_samples, desc="Calculating BLEU scores"):
        if len(sample) <= 10: 
            continue
        split_point = max(10, int(len(sample) * 0.3))
        prompt = sample[:split_point]
        reference = sample[split_point:]
        generated_text = model.prompt(tokenizer, prompt, max_seq_length)
        generated_completion = generated_text[len(prompt):]
        reference_tokens = list(reference)  
        generated_tokens = list(generated_completion)
        try:
            score = sentence_bleu([reference_tokens], generated_tokens, 
                                 weights=(0.25, 0.25, 0.25, 0.25),  
                                 smoothing_function=smooth)
            bleu_scores.append(score)
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            continue
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        return avg_bleu, bleu_scores
    else:
        return 0.0, []

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=128):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        self.text_samples = []
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_obj = json.loads(line)
                    if 'prompt' in json_obj and 'completion' in json_obj:
                        full_text = json_obj['prompt'] + json_obj['completion']
                        self.text_samples.append(full_text)
                except json.JSONDecodeError:
                    print(f"Warning: Couldn't parse JSON line: {line[:50]}...")
        
        print(f"Loaded {len(self.text_samples)} text samples from {file_path}")
        
        self.sequences = []
        for text in self.text_samples:
            tokens = tokenizer.encode(text)
            
            if len(tokens) <= 1:
                continue
                
            if len(tokens) <= seq_length:
                seq = tokens + [tokenizer.eos_id()] 
                if len(seq) >= 2:  
                    self.sequences.append(seq)
            else:
                for i in range(0, len(tokens) - seq_length, seq_length):
                    seq = tokens[i : i + seq_length + 1]
                    self.sequences.append(seq)
            
        print(f"Created {len(self.sequences)} training sequences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        
        output, hidden = self.rnn(embedded, hidden)
        
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits, hidden
    
    def sample_next_token(self, logits):
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1)
        return next_token
    
    def sample_with_temperature(self, logits, temperature=0.7):
        last_logits = logits[:, -1, :]
        scaled_logits = last_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).squeeze(-1)
        return next_token
    
    def prompt(self, tokenizer, text, max_seq_length=100):
        self.eval()
        device = next(self.parameters()).device
        
        input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
        
        generated_tokens = []
        eos_token_id = 3  
        
        with torch.no_grad():
            hidden = None
            
            for _ in range(max_seq_length):
                logits, hidden = self(input_ids, hidden)
                next_token = self.sample_with_temperature(logits, temperature=0.8)
                
                if next_token.item() == eos_token_id:
                    break
                    
                generated_tokens.append(next_token.item())
                
                next_input = next_token.view(1, 1).to(device)

                logits, hidden = self(next_input, hidden)
        
        generated_text = tokenizer.decode(generated_tokens)
        return text + generated_text

def train_rnn(model, train_loader, val_loader, epochs=30, lr=0.001, clip=1.0, device='cuda', patience=3, use_compile=False):
    model.to(device)
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Using compiled model")
        except Exception as e:
            print(f"Model compilation failed, falling back to eager mode: {e}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device != 'cpu'))
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_train_tokens = 0 
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                output, _ = model(inputs)
                output = output.view(-1, model.vocab_size)
                targets = targets.view(-1)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()

            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            scaler.step(optimizer)
            scaler.update()
            non_pad_mask = targets != 0
            total_train_tokens += non_pad_mask.sum().item()
            train_loss += loss.item() * non_pad_mask.sum().item()

            if (batch_idx + 1) % 200 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss = train_loss / total_train_tokens if total_train_tokens > 0 else float('inf')
        train_perplexity = math.exp(train_loss)
        train_losses.append(train_loss)
        
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience * 2:
            print(f"Early stopping after {epoch+1} epochs")
            break
    model.load_state_dict(best_model)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_loss.png')
    plt.close()
    
    return model, train_losses, val_losses

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            output, _ = model(inputs)
            
            output = output.view(-1, model.vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(output, targets)
            
            non_pad_mask = targets != 0
            total_tokens += non_pad_mask.sum().item()
            total_loss += loss.item() * non_pad_mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer_path = "output/my_tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    batch_size = 128
    seq_length = 128
    embedding_dim = 512
    hidden_dim = 512
    num_layers = 2
    dropout = 0.2
    epochs = 30
    lr = 0.001

    train_file = "data/train.jsonl"
    test_file = "data/test.jsonl"
    
    train_dataset = TextDataset(train_file, sp, seq_length)
    test_dataset = TextDataset(test_file, sp, seq_length)
    test_samples = test_dataset.text_samples[:100]  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    vocab_size = sp.get_piece_size()
    rnn_model = RNNModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    rnn_model, train_losses, val_losses = train_rnn(
        rnn_model, train_loader, test_loader,
        epochs=epochs, lr=lr, device=device,
        use_compile=False
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    val_loss, val_ppl = evaluate(rnn_model, test_loader, criterion, device=device)
    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Perplexity: {val_ppl:.4f}")

    avg_bleu, _ = calculate_bleu(rnn_model, sp, test_samples, device=device)
    print(f"Average BLEU score: {avg_bleu:.4f}")
    torch.save(rnn_model.state_dict(), "rnn_model.pt")
    print("\nSample text generation:")
    
    prompt1 = "Which do you prefer? Dogs or cats?"
    generated1 = rnn_model.prompt(sp, prompt1, max_seq_length=100)
    print(f"\nPrompt: {prompt1}")
    print(f"Generated: {generated1}")


if __name__ == "__main__":
    main()
