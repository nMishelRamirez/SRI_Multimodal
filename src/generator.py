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
    """
    Retorna la instancia del cliente OpenAI cargado.
    
    Si el cliente no ha sido inicializado, lo crea utilizando la clave API de OpenAI.
    
    Returns:
        OpenAI: Cliente de la API de OpenAI configurado con la clave API.
        
    Raises:
        ValueError: Si la clave API no está configurada en `src/config.py`.
    """
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no está configurada en src/config.py.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("Cliente OpenAI inicializado.")
    return _openai_client

def build_prompt(query: str, retrieved_captions: List[str]) -> str:
    """
    Construye un prompt para el modelo generativo utilizando la consulta y las descripciones (captions) recuperadas.
    
    Args:
        query (str): La pregunta que se desea responder o la instrucción adicional para la descripción.
                     Puede ser una cadena vacía o la descripción por defecto para búsquedas solo por imagen.
        retrieved_captions (List[str]): Lista de descripciones (captions) de las imágenes que se usan como contexto.
        
    Returns:
        str: El prompt completo para el LLM.
    """
    # Unir los documentos recuperados en una sola cadena de contexto
    context = "\n".join(retrieved_captions)
        
    # Plantilla base del prompt: siempre enfocada en describir las imágenes
    prompt = f"""Eres una aplicación de tipo Retrieval-Augmented Generation (RAG) que siempre responde en español.

    Tu tarea principal es redactar una descripción narrativa y coherente, en español, que resuma el contenido general de las imágenes recuperadas.

    Contexto de descripciones de imágenes recuperadas:
    {context}

    """

    if query and query != "Describe las imágenes recuperadas.": 
        prompt += f"""Además, tu descripción debe enfocarse en responder la siguiente pregunta del usuario:
        "{query}"

        """
    
    prompt += "Descripción detallada:"
    return prompt


def generate_response(prompt: str, max_tokens: int = GENERATION_MAX_NEW_TOKENS) -> str:
    """
    Genera una respuesta utilizando el modelo generativo de OpenAI (GPT).
    
    Args:
        prompt (str): El prompt que se enviará al modelo para generar una respuesta.
        max_tokens (int): El número máximo de tokens que el modelo generará en la respuesta.
        
    Returns:
        str: La respuesta generada por el modelo.
        
    Raises:
        Exception: Si hay un error al generar la respuesta con la API de OpenAI.
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
        print(f"Error al generar respuesta con OpenAI API: {e}")
        return "Lo siento, hubo un error al generar la respuesta. Por favor, inténtalo de nuevo más tarde."