from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
CORS(app)

# Load model from Hugging Face
model_id = "Sankeyyyyy/gpt2-epoch1-checkpoint"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
model.to(device)
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"completion": result})

if __name__ == "__main__":
    app.run(debug=True)




# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# app = Flask(__name__)
# CORS(app)

# # Load model and tokenizer
# checkpoint_path = "./gpt2-epoch1-checkpoint"
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
# model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
# model.eval()

# @app.route("/generate", methods=["POST"])
# def generate():
#     prompt = request.json.get("prompt", "")
#     if not prompt.strip():
#         return jsonify({"completion": "❌ Empty prompt received."})

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     # Generate output
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return jsonify({"completion": generated_text})

# if __name__ == "__main__":
#     app.run(debug=True)

#-----------------------------------------------------------------------------------------

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from model import BigramLanguageModel

# app = Flask(__name__)
# CORS(app)

# # Load vocab and model
# with open("vocab.txt", "r", encoding="utf-8") as f:
#     chars = f.read()
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: chars[i] for i in range(len(chars))}
# encode = lambda s: [stoi[c] for c in s if c in stoi]
# decode = lambda l: ''.join([itos[i] for i in l])

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# vocab_size = len(chars)
# model = BigramLanguageModel(vocab_size, n_embd=64, block_size=32, n_head=4, n_layer=4)
# model.load_state_dict(torch.load("codegpt_model.pt", map_location=device))
# model.eval().to(device)

# @app.route("/generate", methods=["POST"])
# def generate():
#     prompt = request.json.get("prompt", "")
#     idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
#     out = model.generate(idx, max_new_tokens=200, block_size=32)
#     decoded = decode(out[0].tolist())
#     return jsonify({"completion": decoded})

# if __name__ == "__main__":
#     app.run(debug=True)
