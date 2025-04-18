import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from rnn_model import (
    collate_fn,
    evaluate,
    calculate_bleu,
    RNNModel,    
)

from lstm_model import (
    collate_fn,
    evaluate_lstm,
    calculate_bleu,
    LSTMModel,
    train_lstm           
)

from transformer_model import (
    collate_fn,
    evaluate_transformer,
    calculate_bleu,
    TransformerModel,
    train_transformer     
)

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

def load_and_evaluate(model_class, model_path, test_loader, test_samples, tokenizer, device='cuda'):
    model = model_class(
        vocab_size=tokenizer.get_piece_size(),
        embedding_dim=512,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2
    )
    model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if isinstance(model, TransformerModel):
        test_loss, test_ppl = evaluate_transformer(model, test_loader, criterion, device=device)
    else:
        test_loss, test_ppl = evaluate(model, test_loader, criterion, device=device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_ppl:.4f}")

    avg_bleu, bleu_scores = calculate_bleu(
        model, tokenizer,
        test_samples,
        max_seq_length=100,
        device=device
    )
    print(f"Average BLEU score: {avg_bleu:.4f}")

    return model, test_loss, test_ppl, avg_bleu, bleu_scores


if __name__ == "__main__":
    import sentencepiece as spm
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sp = spm.SentencePieceProcessor()
    sp.load("output/my_tokenizer.model")

    test_dataset = TextDataset("data/test.jsonl", sp, seq_length=128)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_samples = test_dataset.text_samples[:100]

    print("\nRNN Model Evaluation")
    model, loss, ppl, bleu, bleu_scores = load_and_evaluate(
        RNNModel,          
        "rnn_model.pt",    
        test_loader,
        test_samples,
        sp,
        device=device
    )

    print("\nLSTM Model Evaluation")
    model, loss, ppl, bleu, bleu_scores = load_and_evaluate(
        LSTMModel,          
        "lstm_model.pt",    
        test_loader,
        test_samples,
        sp,
        device=device
    )

    print("\nTransformer Model Evaluation")
    model, loss, ppl, bleu, bleu_scores = load_and_evaluate(
        TransformerModel,          
        "transformer_model.pt",    
        test_loader,
        test_samples,
        sp,
        device=device
    )
    