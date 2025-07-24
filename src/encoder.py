import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_image(img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = model.encode_image(image)
    return img_embedding / img_embedding.norm()

def encode_text(text):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_token)
    return text_embedding / text_embedding.norm()
