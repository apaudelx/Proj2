import os
import argparse
import sentencepiece as spm
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("tokenizer_training.log", encoding='utf-8')
        ]
    )
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            handler.stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    
    return logging.getLogger(__name__)

def process_raw_data(input_files, output_file):
    logger.info(f"Processing {len(input_files)} input files into {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        total_lines = 0
        for input_file in input_files:
            if not os.path.exists(input_file):
                logger.warning(f"File {input_file} does not exist, skipping.")
                continue
                
            logger.info(f"Processing {input_file}")
            with open(input_file, 'r', encoding='utf-8') as inf:
                for line in inf:
                    line = line.strip()
                    if line: 
                        outf.write(line + '\n')
                        total_lines += 1
                        
            logger.info(f"Processed {total_lines} lines so far")
            
    logger.info(f"Finished processing. Total lines: {total_lines}")
    return total_lines

def train_tokenizer(input_file, model_prefix, vocab_size, character_coverage, model_type, input_sentence_size=0):
    logger.info(f"Training {model_type} tokenizer with vocab size {vocab_size}")
    
    train_kwargs = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'model_type': model_type,
        'pad_id': 0,
        'unk_id': 1,
        'bos_id': 2,
        'eos_id': 3,
        'input_sentence_size': input_sentence_size,
        'shuffle_input_sentence': True,
        'normalization_rule_name': 'identity',
        'add_dummy_prefix': True,
        'remove_extra_whitespaces': False
    }
    logger.info("Starting tokenizer training with parameters:")
    logger.info(str(train_kwargs))
    spm.SentencePieceTrainer.Train(**train_kwargs)
    
    logger.info(f"Tokenizer training complete. Model saved to {model_prefix}.model and {model_prefix}.vocab")

def calculate_tokenization_metrics(sp, test_sentences):
    """Calculate metrics to evaluate tokenization quality."""
    total_sentences = len(test_sentences)
    perfect_reconstructions = 0
    total_tokens = 0
    total_unk_tokens = 0
    total_chars = 0
    total_character_similarity = 0
    
    logger.info("\n===== Tokenization Metrics =====")
    
    for sentence in test_sentences:
        original_chars = len(sentence)
        tokens = sp.encode(sentence, out_type=str)
        token_ids = sp.encode(sentence, out_type=int)
        total_tokens += len(tokens)
        unk_count = token_ids.count(1)
        total_unk_tokens += unk_count
        token_chars = sum(len(t.replace('▁', '')) for t in tokens)
        total_chars += token_chars
        decoded = sp.decode(token_ids)
        is_perfect = (decoded.lower() == sentence.lower())
        if is_perfect:
            perfect_reconstructions += 1
        import difflib
        character_similarity = difflib.SequenceMatcher(None, sentence.lower(), decoded.lower()).ratio() * 100
        total_character_similarity += character_similarity
    reconstruction_accuracy = (perfect_reconstructions / total_sentences) * 100
    token_coverage = ((total_tokens - total_unk_tokens) / total_tokens) * 100 if total_tokens > 0 else 0
    avg_token_length = total_chars / total_tokens if total_tokens > 0 else 0
    avg_character_similarity = total_character_similarity / total_sentences
    logger.info(f"Reconstruction Accuracy (case-insensitive): {reconstruction_accuracy:.2f}% ({perfect_reconstructions}/{total_sentences} sentences)")
    logger.info(f"Average Character Similarity: {avg_character_similarity:.2f}%")
    logger.info(f"Token Coverage: {token_coverage:.2f}% (UNK tokens: {total_unk_tokens}/{total_tokens})")
    logger.info(f"Average Token Length: {avg_token_length:.2f} characters/token")
    try:
        with open(os.path.join("data", "validation.txt"), "r", encoding="utf-8") as f:
            validation_text = f.read()
            logger.info("\nValidation Set Metrics:")
            if len(validation_text) > 1000:
                validation_sample = validation_text[:1000]
            else:
                validation_sample = validation_text
                
            val_tokens = sp.encode(validation_sample, out_type=int)
            val_unk_count = val_tokens.count(1)
            val_coverage = ((len(val_tokens) - val_unk_count) / len(val_tokens)) * 100 if len(val_tokens) > 0 else 0
            logger.info(f"Validation Token Coverage: {val_coverage:.2f}%")
    except FileNotFoundError:
        logger.info("No validation file found. Skipping validation metrics.")
    
    return {
        "reconstruction_accuracy": reconstruction_accuracy,
        "character_similarity": avg_character_similarity,
        "token_coverage": token_coverage,
        "avg_token_length": avg_token_length
    }

def test_tokenizer(model_path, test_sentences):
    logger.info(f"Testing tokenizer {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    logger.info(f"Vocabulary size: {sp.get_piece_size()}")
    logger.info(f"PAD token ID: 0")
    logger.info(f"UNK token ID: 1")
    logger.info(f"BOS token ID: 2")
    logger.info(f"EOS token ID: 3")
    logger.info("\nTokenization examples:")
    for sentence in test_sentences:
        token_ids = sp.encode(sentence, out_type=int)
        try:
            tokens = sp.encode(sentence, out_type=str)
            logger.info(f"Original: {sentence}")
            logger.info(f"Tokens: {tokens}")
        except UnicodeEncodeError:
            tokens = sp.encode(sentence, out_type=str)
            logger.info(f"Original: {sentence}")
            logger.info(f"Tokens (encoded): {[repr(t) for t in tokens]}")
        decoded = sp.decode(token_ids)
        logger.info(f"Token IDs: {token_ids}")
        logger.info(f"Decoded: {decoded}")
        logger.info("-" * 50)
    metrics = calculate_tokenization_metrics(sp, test_sentences)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer for language modeling")
    parser.add_argument("--input_dir", type=str, default="data/raw", 
                        help="Directory containing input text files for training the tokenizer")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="Directory to save the trained tokenizer model")
    parser.add_argument("--model_name", type=str, default="tokenizer", 
                        help="Name of the tokenizer model")
    parser.add_argument("--vocab_size", type=int, default=10000, 
                        help="Size of the vocabulary")
    parser.add_argument("--character_coverage", type=float, default=0.9995, 
                        help="Character coverage, usually 0.9995 for English, 0.9999 for other languages")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram", "char", "word"],
                        help="Model type (bpe, unigram, char, or word)")
    parser.add_argument("--max_sentences", type=int, default=0, 
                        help="Maximum number of sentences to use for training (0 means all)")
    parser.add_argument("--random_seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    import random
    random.seed(args.random_seed)

    import glob 
    input_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not input_files:
        logger.error(f"No .txt files found in directory {args.input_dir}")
        return
    combined_input_file = os.path.join(args.output_dir, "combined_input.txt")
    process_raw_data(input_files, combined_input_file)
    model_prefix = os.path.join(args.output_dir, args.model_name)
    train_tokenizer(
        input_file=combined_input_file,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=args.max_sentences
    )
    test_sentences = [
        "Hello, this is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks have revolutionized natural language processing.",
        "SentencePiece is an unsupervised text tokenizer and detokenizer.",
        "In machine learning, we often need to represent text as numerical vectors."
    ]
    print(test_tokenizer(model_prefix + ".model", test_sentences))
        
    if len(input_files) > 1:  
        logger.info(f"Cleaning up temporary file {combined_input_file}")
        os.remove(combined_input_file)

    logger.info("Tokenizer training and testing complete!")

if __name__ == "__main__":
    logger = setup_logging()
    main()