import torch
import tiktoken

from GPT import GPT2
from configs import GPTConfig

model_path = "E004/E004-latest_checkpoint.pt"
checkpoint = torch.load(model_path,map_location="cpu")

model = GPT2(GPTConfig())
state_dict = checkpoint['model']

# Remove torch.compile prefix
state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in state_dict.items()
}

model.load_state_dict(state_dict)
model.to("cpu")
model.eval()

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

prompt = "Amy had always dreamed of becoming a great artist."

idx = torch.tensor([encode(prompt)],dtype=torch.long).to("cpu")

generated = model.generate(
    idx,
    max_new_tokens=1500,
    temperature=0.7,
    top_k=40
)
print(decode(generated[0].tolist()))
