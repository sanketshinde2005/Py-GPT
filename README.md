# Py-GPT: NanoGPT-based Python Code Generator

## Overview

**Py-GPT** is a project that leverages a custom-trained GPT model to generate Python code from function signatures. The project is organized into three main parts:

- **Backend:** Python API serving the GPT model for code generation.
- **Frontend:** Next.js web interface for user interaction.
- **Notebooks:** Jupyter notebooks for model training, tokenization, and experimentation.

---

## Project Structure

```
NanoGPT/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── [model files, checkpoints, configs, vocab, etc.]
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

## Notebooks

### 1. Py-GPT (High level Tokenization).ipynb

- **Purpose:**  
  Demonstrates high-level tokenization strategies for preparing Python code data for GPT training.
- **Contents:**  
  - Data preprocessing and cleaning
  - Tokenization using HuggingFace or custom tokenizers
  - Exploratory data analysis on token distributions
  - Saving processed datasets for model training

### 2. Py-GPT (Low level Tokenization).ipynb

- **Purpose:**  
  Explores low-level, character-based or byte-level tokenization for fine-grained control.
- **Contents:**  
  - Implementation of custom tokenization logic
  - Comparison with high-level tokenization
  - Visualization of token sequences
  - Preparing data for low-level model training

---

## Backend

### Overview

- **API:**  
  The backend is a Python FastAPI/Flask app (`app.py`) that loads the trained GPT model and exposes an endpoint for code generation.
- **Model Files:**  
  - Model weights (`.pt`, `.safetensors`, etc.)
  - Tokenizer files (`vocab.json`, `merges.txt`, etc.)
  - Config files for model architecture and generation settings

### Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the API server:**
   ```sh
   python app.py
   ```
3. **API Endpoint:**
   - `POST /generate`  
     Request: `{ "prompt": "def my_function(x):" }`  
     Response: `{ "completion": "def my_function(x):\n    # generated code..." }`

---

## Frontend

### Overview

- **Framework:** Next.js (React + TypeScript)
- **Features:**
  - User inputs a Python function signature
  - Sends request to backend and displays generated code
  - Light/Dark mode toggle
  - Copy-to-clipboard functionality

### Usage

1. **Install dependencies:**
   ```sh
   cd frontend
   npm install
   ```
2. **Run the development server:**
   ```sh
   npm run dev
   ```
3. **Access the app:**  
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Model Training & Customization

- **Training:**  
  Use the provided notebooks to preprocess data and train your own GPT model.
- **Customization:**  
  - Adjust tokenization strategies in the notebooks.
  - Fine-tune model parameters and architecture as needed.
- **Deployment:**  
  Place your trained model files in the `backend` directory and update `app.py` to load them.

---

## Requirements

- **Backend:** Python 3.8+, PyTorch, Transformers, Flask/FastAPI
- **Frontend:** Node.js 18+, npm
- **Notebooks:** Jupyter, pandas, numpy, matplotlib, transformers

---

## Notes

- **Large files:**  
  Model weights and checkpoints are not tracked by Git. Use [Git LFS](https://git-lfs.github.com/) if you need to version large files.
- **.gitignore:**  
  The project is set up to ignore Python cache, model files, node_modules, build artifacts, and environment files.

---

## License

MIT License

---

## Acknowledgements

- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
-
