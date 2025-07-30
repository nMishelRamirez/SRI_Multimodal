# SRI_Multimodal/src/encoder.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os

from config import CLIP_MODEL_NAME, get_device

class MultimodalEncoder:
    """
    Clase para codificar imágenes y texto en embeddings utilizando un modelo CLIP.
    """
    def __init__(self, model_name=CLIP_MODEL_NAME):
        self.device = get_device()
        print(f"Cargando modelo CLIP ({model_name}) en dispositivo: {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("Modelo CLIP cargado exitosamente.")

    def encode_image(self, image_path):
        """Codifica una imagen en un embedding vectorial."""
        if not os.path.exists(image_path):
            print(f"Error: La imagen no se encontró en la ruta: {image_path}")
            return None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error al cargar o procesar imagen {image_path}: {e}")
            return None
        
        # El procesador de CLIP ya maneja el preprocesamiento necesario
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(): # Desactivar el cálculo de gradientes para inferencia
            image_features = self.model.get_image_features(**inputs)
        # Mover a CPU y convertir a numpy para compatibilidad con FAISS y otras operaciones
        return image_features.cpu().numpy()

    def encode_text(self, text):
        """Codifica un texto en un embedding vectorial."""
        if not isinstance(text, str) or not text.strip():
            print("Advertencia: Se intentó codificar texto vacío o no válido.")
            return np.array([]) # Retorna un array vacío o un valor que indique error

        # El procesador de CLIP ya maneja el preprocesamiento necesario
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()