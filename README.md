# Py-GPT: NanoGPT-based Python Code Generator

## Overview

**Py-GPT** is a project that leverages a custom-trained GPT-2 language model to generate Python code from function signatures. The project is organized into three main components:

- **Backend:** Python Flask API serving the GPT model for code generation.
- **Frontend:** Next.js web interface for user interaction.
- **Notebooks:** Jupyter notebooks for model training, tokenization, and experimentation.

---

## 🚀 Hosted Model on Hugging Face

Trained model files are hosted on Hugging Face and automatically loaded in the backend.

🔗 Hugging Face Repo:  
https://huggingface.co/Sankeyyyyy/gpt2-epoch1-checkpoint

To download and load the model during runtime:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("Sankeyyyyy/gpt2-epoch1-checkpoint")
tokenizer = GPT2Tokenizer.from_pretrained("Sankeyyyyy/gpt2-epoch1-checkpoint")
```

---

## Project Structure

```
NanoGPT/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── (Model is loaded from Hugging Face)
├── frontend/
│   ├── .gitignore
│   ├── package.json
│   ├── next.config.ts
│   ├── public/
│   └── src/
├── notebook/
│   ├── Py-GPT (High level Tokenization).ipynb
│   └── Py-GPT (Low level Tokenization).ipynb
└── README.md
```

---

## 🧠 Model Architecture

- **Base Model:** GPT2 (from HuggingFace `gpt2`)
- **Fine-tuning:**
  - Trained on a custom dataset of Python function signatures and implementations.
  - Used Hugging Face’s `Trainer` class with `transformers` library.
  - Training run completed for 1 epoch (with resume support for more).

**Key Hyperparameters:**
- Block Size: 256
- Batch Size: 2
- Epochs: 3 (in progress)
- Optimizer: AdamW
- Evaluation Strategy: `epoch`
- Tokenizer: `GPT2Tokenizer` with `eos_token` as padding

---

## 🧪 Notebooks

### 1. Py-GPT (High level Tokenization)

- Tokenization using `transformers.GPT2Tokenizer`
- Preprocessing with Pandas, Datasets
- Prepares HuggingFace-compatible dataset for training

### 2. Py-GPT (Low level Tokenization)

- Custom character/byte-level tokenization using vocab files
- Inspired by `NanoGPT` architecture
- Used with manually defined `BigramLanguageModel`

---

## 🖥️ Backend

### Overview

- **API:** Flask app (`app.py`) that serves the trained model via an endpoint.
- **Model Source:** Loaded from Hugging Face Hub.

### Usage

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### API Endpoint

`POST /generate`

- **Request:**
  ```json
  { "prompt": "def binary_search(arr, target):" }
  ```

- **Response:**
  ```json
  { "completion": "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    ..." }
  ```

---

## 🌐 Frontend

### Overview

- Built using **Next.js** with TypeScript
- UI allows users to enter function prompts and view generated code
- Features:
  - Light/Dark theme toggle
  - Copy to clipboard
  - Loading animation

### Usage

```bash
cd frontend
npm install
npm run dev
```

Then visit: [http://localhost:3000](http://localhost:3000)

---

## 🧠 Training Details

- **Library:** Hugging Face Transformers
- **Trainer:** `transformers.Trainer`
- **Dataset:** Custom Python functions in CSV format
- **Colab/Local Training:** Compatible with both Google Colab and Jupyter

---

## 📦 Requirements

- **Backend:** Python 3.8+, Flask, Torch, Transformers
- **Frontend:** Node.js 18+, npm
- **Training:** Jupyter, pandas, transformers, datasets

---

## ❗ Notes

- **Model files >100MB** are stored on Hugging Face instead of GitHub.
- Use Git LFS only if hosting models locally:
  ```bash
  git lfs install
  git lfs track "*.pt" "*.safetensors"
  ```

---

## 📜 License

MIT License

---

## Acknowledgements

- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI GPT-2](https://openai.com/research/better-language-models)

---

> ⭐ If you like this project, consider giving it a star on GitHub!
