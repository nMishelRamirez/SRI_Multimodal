import openai
import pandas as pd
from app.config import OPENAI_API_KEY, CONCEPTS_FILE

openai.api_key = OPENAI_API_KEY

# Cargar corpus 2
concept_df = pd.read_csv(CONCEPTS_FILE)
concept_dict = dict(zip(concept_df["label"], concept_df["description"]))

def build_context_from_labels(labels):
    context = ""
    for label in labels:
        desc = concept_dict.get(label.strip().lower())
        if desc:
            context += desc + "\n"
    return context

def generate_response(context, query):
    prompt = f"""
    Contexto recuperado:

    {context}

    Consulta del usuario: {query}

    Redacta una respuesta explicativa o narrativa sobre el tema.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
