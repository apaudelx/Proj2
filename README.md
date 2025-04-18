# Text Generation Project

This project implements and evaluates **RNN**, **LSTM**, and **Transformer** models for text generation and evaluation using PyTorch.

## Getting Started

Follow the steps below to run the key components of this project.

---

### 1. Train the Tokenizer

Train a SentencePiece tokenizer on raw text data:

```bash
python train_tokenizer.py \
  --input_dir data/raw \
  --output_dir output \
  --model_name my_tokenizer \
  --vocab_size 10000
```

---

### 2. Train Models

Train one of the sequential models (RNN, LSTM, or Transformer) using the preprocessed data and trained tokenizer.

- **RNN Model**

  ```bash
  python rnn_model.py
  ```

- **LSTM Model**

  ```bash
  python lstm_model.py
  ```

- **Transformer Model**

  ```bash
  python transformer_model.py
  ```

---

### 3. Evaluate Models

Evaluate the trained models on the test dataset:

```bash
python eval.py
```

---

### 4. Generate Text

Generate sample text using the trained models:

- **RNN**

  ```bash
  python test_rnn.py
  ```
  To use a custom prompt:
  ```bash
  python test_rnn.py --prompt "Your custom prompt here"
  ```

- **LSTM**

  ```bash
  python test_lstm.py
  ```
  To use a custom prompt:
  ```bash
  python test_lstm.py --prompt "Your custom prompt here"
  ```

- **Transformer**

  ```bash
  python test_transformer.py
  ```
  To use a custom prompt:
  ```bash
  python test_transformer.py --prompt "Your custom prompt here"
  ```
---

## Notes
- The `env.yaml` file specifies all the dependencies required to run this project. You can use it to set up a conda environment with the following commands:
  ```bash
  conda env create -f env.yaml
  conda activate text-gen-env
  ```
- Once the environment is set up, follow the instructions above to train, evaluate, and generate text using the models.
- To quickly test the project, refer to the **Generate Text** section.
