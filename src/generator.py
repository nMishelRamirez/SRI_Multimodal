# SRI_Multimodal/src/generator.py

from openai import OpenAI
from typing import List
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importaciones absolutas desde el paquete src
from src.config import OPENAI_API_KEY, GENERATIVE_MODEL_NAME, GENERATION_MAX_NEW_TOKENS

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
    

def build_prompt(text_query: str, retrieved_captions: List[str]) -> str:
    """
    Construye un prompt para el modelo generativo utilizando la consulta y las descripciones (captions) recuperadas.
    El prompt está diseñado para generar una descripción narrativa coherente del contenido de las imágenes.
    
    Args:
        query (str): La pregunta o instrucción del usuario. Si es una búsqueda por imagen sin texto adicional,
                     se espera una cadena genérica como "Describe las imágenes recuperadas.".
        retrieved_captions (List[str]): Lista de descripciones (captions) de las imágenes recuperadas,
                                         que sirven como contexto para el LLM.
        
    Returns:
        str: El prompt completo para el LLM.
    """
    # Unir las descripciones recuperadas para formar el contexto
    context = "\n".join([f"- {caption}" for caption in retrieved_captions])

    prompt = f"""Eres un asistente que describe imágenes de forma clara, narrativa y en español.

    Reglas estrictas:
    1. Tu respuesta debe contener **solo la descripción**, sin frases introductorias como "Claro," o "Aquí está...,".
    2. **No hagas referencia** a descripciones previas, fuentes de texto, subtítulos o la consulta.
    3. **Comienza directamente describiendo la imagen**, por ejemplo: "En esta imagen se observa..."
    4. Unicamente responde con **UNA** descripcion general, No redactes mas de una descripción.
    5. No inventes información que no esté en el contexto.
    6. Si la consulta es ambigua (ej. "¿Qué es esto?"), responde según la **primera descripción**.
    7. Si no hay información suficiente, responde literalmente:
    "Lo siento, no encontré información suficiente para responder a tu consulta."

    ---
    Contexto de las imágenes recuperadas:
    {context}

    Pregunta del usuario:
    {text_query}
    """
    return prompt


def generate_response(prompt: str, client, modelo: str = "gpt-4.1") -> str:
    try:
        response = client.responses.create(
            model=modelo,
            input=prompt
        )
        return  response.output[0].content[0].text
    
    except Exception as e:
        print(f"Error al generar respuesta con OpenAI API: {e}")
        return "Lo siento, hubo un error al generar la respuesta. Por favor, inténtalo de nuevo más tarde."
