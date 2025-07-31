# SRI_Multimodal/src/preprocessing.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from typing import List

# Recursos de NLTK 
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4') 
# nltk.download('punkt_tab')

# Inicializar recursos de NLTK
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("Advertencia: Recursos de NLTK no encontrados. Ejecuta 'python -c \"import nltk; nltk.download(\'punkt\'); nltk.download(\'stopwords\'); nltk.download(\'wordnet\'); nltk.download(\'omw-1.4\')\"'")
    stop_words = set() # Fallback si no están descargados
    lemmatizer = None

def clean_text(text: str) -> str:
    """Elimina caracteres especiales, números y convierte a minúsculas."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)                             # Eliminar texto entre corchetes
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)   # Eliminar puntuación
    text = re.sub(r'\d+', '', text)                                 # Eliminar números
    text = re.sub(r'\s+', ' ', text).strip()                        # Eliminar espacios extra
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokeniza el texto en palabras."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Elimina las stopwords de una lista de tokens."""
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lematiza una lista de tokens."""
    if lemmatizer:
        return [lemmatizer.lemmatize(word) for word in tokens]
    return tokens  # Retorna sin lematizar si lemmatizer no está disponible

def preprocess_document(document: str) -> str:
    """Aplica todas las etapas de preprocesamiento a un solo documento."""
    cleaned_doc = clean_text(document)
    tokens = tokenize_text(cleaned_doc)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    return ' '.join(lemmatized_tokens)

def preprocess_documents(documents: List[str]) -> pd.DataFrame:
    """
    Aplica el preprocesamiento a una lista de documentos y retorna un DataFrame.
    """
    preprocessed_docs = [preprocess_document(doc) for doc in documents]
    df_processed = pd.DataFrame({'original_doc': documents, 'prep_doc': preprocessed_docs})
    return df_processed

def concatenar_captions_by_image(df: pd.DataFrame, image_col: str = 'file_name', caption_col: str = 'caption') -> pd.DataFrame:
    """
    Combina múltiples captions para la misma imagen en una sola cadena de texto.
    """
    df_combined = df.groupby(image_col)[caption_col].apply(lambda x: ' '.join(x)).reset_index()
    df_combined.rename(columns={caption_col: 'combined_caption'}, inplace=True)
    return df_combined
