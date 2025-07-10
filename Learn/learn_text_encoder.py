from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import torch
from pathlib import Path

encoder_path1 = Path("E:/_PosterMaker/PosterMaker/checkpoints/stable-diffusion-3-medium-diffusers/text_encoder")
encoder_path2 = Path("E:/_PosterMaker/PosterMaker/checkpoints/stable-diffusion-3-medium-diffusers/tokenizer")

tokenizer = CLIPTokenizer.from_pretrained(encoder_path2, local_files_only=True)
model = CLIPTextModelWithProjection.from_pretrained(encoder_path1, local_files_only=True)

text = "a surreal painting of a futuristic city with floating cars"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

print(tokenizer.tokenize(text))
print(inputs["input_ids"])

with torch.no_grad():
    embedding = model(**inputs).last_hidden_state

print("Embedding shape:", embedding.shape)
