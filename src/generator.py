# SRI_Multimodal/src/generator.py

from openai import OpenAI
from typing import List
# Importación relativa para config
from src.config import OPENAI_API_KEY, GENERATIVE_MODEL_NAME

# Inicializar cliente OpenAI
_openai_client = None

def get_openai_client():
    """Retorna la instancia del cliente OpenAI cargado."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no está configurada en src/config.py.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("Cliente OpenAI inicializado.")
    return _openai_client

def build_prompt(query: str, retrieved_captions: List[str]) -> str:
    """
    Construye un prompt para el modelo generativo usando la consulta y los captions recuperados.
    """
    context = "\n".join([f"- {caption}" for caption in retrieved_captions])
    prompt = f"Basado en las siguientes descripciones de imágenes:\n{context}\n\nResponde la siguiente pregunta: {query}\nRespuesta:"
    return prompt

def generate_response(prompt: str, max_tokens: int = 200) -> str:
    """
    Genera una respuesta utilizando el modelo de lenguaje de OpenAI.
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=GENERATIVE_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un asistente útil que describe imágenes y responde preguntas basadas en descripciones proporcionadas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al generar respuesta con OpenAI: {e}")
        return f"Lo siento, no pude generar una respuesta. Error: {e}"