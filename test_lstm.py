import torch
import argparse
import sentencepiece as spm
from lstm_model import LSTMModel  

def test_existing_model(prompt=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer_path = "output/my_tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    vocab_size = sp.get_piece_size()
    lstm_model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=512,   
        hidden_dim=512,      
        num_layers=2,        
        dropout=0.2          
    )
    
    print("Loading saved model weights from lstm_model.pt...")
    try:
        lstm_model.load_state_dict(torch.load("lstm_model.pt"))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    lstm_model.to(device)
    lstm_model.eval()  
    
    if prompt:
        prompts = [prompt]
    else:
        prompts = [
            "Which do you prefer? Dogs or cats?",
            "Artificial intelligence is"
        ]
    
    temperatures = [0.5, 0.8, 1.2]
    
    print("\n===== TESTING LSTM TEXT GENERATION =====")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        for temp in temperatures:
            generated = generate_with_temp(lstm_model, sp, prompt, temperature=temp, max_seq_length=100)
            print(f"\nTemperature {temp}:")
            print(f"{generated}")
        
        print("\n" + "-"*60)

def generate_with_temp(model, tokenizer, text, temperature=0.8, max_seq_length=100):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
    
    generated_tokens = []
    eos_token_id = 3  
    
    with torch.no_grad():
        hidden = None
        
        logits, hidden = model(input_ids, hidden)
        
        for _ in range(max_seq_length):
            next_token = model.sample_with_temperature(logits, temperature)
            
            if next_token.item() == eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            
            next_input = next_token.view(1, 1).to(device)
            
            logits, hidden = model(next_input, hidden)
        
    generated_text = tokenizer.decode(generated_tokens)
    return text + generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LSTM text generation.")
    parser.add_argument("--prompt", type=str, help="Input prompt for text generation.")
    args = parser.parse_args()
    
    test_existing_model(prompt=args.prompt)