import torch
import argparse
import sentencepiece as spm
from transformer_model import TransformerModel

def test_existing_transformer(prompt=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer_path = "output/my_tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    vocab_size = sp.get_piece_size()
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=512,   
        hidden_dim=512,      
        nhead=2,             
        num_layers=2,        
        dropout=0.2,         
        max_seq_length=512   
    )
    
    print("Loading saved model weights from transformer_model.pt...")
    try:
        transformer_model.load_state_dict(torch.load("transformer_model.pt", map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    transformer_model.to(device)
    transformer_model.eval()  
    
    if prompt:
        prompts = [prompt]
    else:
        prompts = [
            "Which do you prefer? Dogs or cats?",
            "Artificial intelligence is"
        ]
    
    temperatures = [0.5, 0.8, 1.2]
    
    print("\n===== TESTING TRANSFORMER TEXT GENERATION =====")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        for temp in temperatures:
            generated = generate_with_temp(transformer_model, sp, prompt, temperature=temp, max_seq_length=100)
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
        for _ in range(max_seq_length):
            if input_ids.size(1) > model.max_seq_length:
                input_ids = input_ids[:, -model.max_seq_length:]
            
            output = model(input_ids)
            
            next_token = model.sample_with_temperature(output, temperature)
            
            if next_token.item() == eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            
            next_token = next_token.view(1, 1)  
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = tokenizer.decode(generated_tokens)
    return text + generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Transformer text generation.")
    parser.add_argument("--prompt", type=str, help="Input prompt for text generation.")
    args = parser.parse_args()
    
    test_existing_transformer(prompt=args.prompt)