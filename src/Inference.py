import torch
import tiktoken

import GenerationUtils as GU
from GPT_Block.GPT import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    model_path = "Model/model_v2.pth"

    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    start_context = "I have fresh food"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = GU.generate(
        model=model,
        idx=GU.text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", GU.token_ids_to_text(token_ids, tokenizer))


# (LLM-venv) [sheikhmadni@fedora /data/ML/DL/LLMs-From-Scratch]
# ↳ $ python src/Inference.py 
# Output text:
#  Every effort moves you?"
# "Yes--quite insensible to
# (LLM-venv) [sheikhmadni@fedora /data/ML/DL/LLMs-From-Scratch]
# ↳ $ python src/Inference.py 
# Output text:
#  I am Madni and I meant to do the picture for nothing--I
# (LLM-venv) [sheikhmadni@fedora /data/ML/DL/LLMs-From-Scratch]
# ↳ $ python src/Inference.py 
# Output text:
#  I have fresh food I had been through--forming, as it--