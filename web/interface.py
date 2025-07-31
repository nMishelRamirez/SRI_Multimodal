import sys
import os

import streamlit as st
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importaciones absolutas desde el paquete src
from src.config import FLICKR8K_DATASET_DIR, TOP_K_RETRIEVAL, GENERATIVE_MODEL_NAME
from src.embedding import generate_image_embedding, generate_text_embeddings
from src.indexer import FaissVectorDB # Renombrado de vector_db
from src.retriever import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image # Renombrado de search_engine
from src.generator import build_prompt, generate_response # Renombrado de generation

@st.cache_resource
def load_all_resources():
    """
    Carga todos los recursos necesarios para la aplicación, incluyendo la base de datos FAISS.
    Se utiliza st.cache_resource para que se cargue una única vez.
    """
    st.write("Cargando base de datos FAISS...")
    faiss_db_instance = FaissVectorDB.load_index() # load_index ya usa las rutas de config
    if faiss_db_instance is None:
        st.error("Error al cargar el índice FAISS. Por favor, asegúrate de haber ejecutado 'python src/build_index.py' primero.")
        st.stop() # Detiene la ejecución de la app Streamlit
        
    return faiss_db_instance

# Cargar todos los recursos
faiss_db = load_all_resources()

# Configuración de la página Streamlit
st.set_page_config(layout="wide", page_title="SRI Multimodal RAG")
st.title("Sistema de Recuperación Multimodal de Información (SRI-RAG)")

# --- Sección de Entrada de Consulta ---
st.header("1. Ingresa tu Consulta")

query_input_text = st.text_input("Ingresa tu consulta de texto aquí (ej. 'a dog playing in the grass')", "")
uploaded_image = st.file_uploader("O sube una imagen aquí", type=["jpg", "jpeg", "png"])

search_button = st.button("Realizar Búsqueda") # Un solo botón para ambas búsquedas

# --- Sección de Resultados de Búsqueda ---
st.header("2. Resultados de la Búsqueda")
retrieved_results = []

if search_button:
    if query_input_text and uploaded_image:
        st.info(f"Realizando búsqueda combinada para: '{query_input_text}' y imagen '{uploaded_image.name}'...")
        # Guardar la imagen temporalmente para poder usar su ruta
        temp_image_path = os.path.join("temp_uploaded_image.jpg") # Puedes ajustar la ruta temporal
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        distances, indices = retrieve_by_text_and_image(query_input_text, temp_image_path, faiss_db, k=TOP_K_RETRIEVAL)
        os.remove(temp_image_path) # Eliminar la imagen temporal
        
    elif query_input_text:
        st.info(f"Realizando búsqueda por texto para: '{query_input_text}'...")
        distances, indices = retrieve_by_text(query_input_text, faiss_db, k=TOP_K_RETRIEVAL)
        
    elif uploaded_image:
        st.info(f"Realizando búsqueda por imagen para: '{uploaded_image.name}'...")
        # Guardar la imagen temporalmente
        temp_image_path = os.path.join("temp_uploaded_image.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
            
        distances, indices = retrieve_by_image(temp_image_path, faiss_db, k=TOP_K_RETRIEVAL)
        os.remove(temp_image_path) # Eliminar la imagen temporal
        
    else:
        st.warning("Por favor, ingresa una consulta de texto o sube una imagen para buscar.")
        indices = None # Asegurar que indices sea None para no entrar al bucle si no hay consulta

    if indices is not None and len(indices[0]) > 0:
        # Filtrar índices -1 y obtener información de los resultados válidos
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(faiss_db.image_info)]
        # Ordenar los resultados por distancia (FAISS ya los devuelve ordenados, pero es una buena práctica)
        # Asegurarse de que las distancias también se correspondan con los índices válidos
        valid_distances = [d for i, d in enumerate(distances[0]) if indices[0][i] != -1 and indices[0][i] < len(faiss_db.image_info)]
        
        # Crear la lista de resultados con sus metadatos
        for i, idx in enumerate(valid_indices):
            info = faiss_db.image_info[idx]
            retrieved_results.append({
                'file_name': info['file_name'],
                'image_path': info['image_path'],
                'caption': info['caption'],
                'distance': valid_distances[i] # Incluir la distancia
            })
        
        # Mostrar resultados con la nueva disposición
        if retrieved_results:
            st.write(f"Se recuperaron {len(retrieved_results)} resultados.")

            # Separar el Top-1 del resto
            top_1_result = retrieved_results[0]
            other_results = retrieved_results[1:]

            # Columna principal para el Top-1
            col1, col2 = st.columns([0.45, 0.55]) 

            with col1:
                st.subheader("Top-1 Resultado")
                try:
                    # Mostrar la imagen más grande, sin ID ni descripción
                    st.image(Image.open(top_1_result['image_path']).convert("RGB"), width=400)
                except Exception as e:
                    st.error(f"Error al cargar la imagen principal {top_1_result['image_path']}: {e}")

            with col2:
                st.subheader("Otros Resultados")
                # Organizar las otras imágenes en filas, 2 por fila
                num_other_results = len(other_results)
                for i in range(0, num_other_results, 2):
                    row_cols = st.columns(2)
                    for j in range(2):
                        if (i + j) < num_other_results:
                            res = other_results[i + j]
                            with row_cols[j]:
                                try:
                                    # Mostrar imágenes más pequeñas, sin ID ni descripción
                                    st.image(Image.open(res['image_path']).convert("RGB"), width=150)
                                    # print(f"DEBUG - Other Image Path: {res['image_path']}")
                                    # print(f"DEBUG - Other Caption: {res['caption']}")
                                except Exception as e:
                                    st.error(f"Error al cargar imagen {res['image_path']}: {e}")
            
            # --- Generación Aumentada por Recuperación (RAG) ---
            st.header("3. Respuesta Generada (RAG)")
            # Extraer solo los captions para el RAG
            context_for_rag = [res['caption'] for res in retrieved_results]

            if context_for_rag:
                final_query_for_llm = query_input_text if query_input_text else "Describe las imágenes recuperadas."
                
                with st.spinner(f"Generando respuesta narrativa/explicativa con {GENERATIVE_MODEL_NAME}..."):
                    rag_prompt = build_prompt(final_query_for_llm, context_for_rag)
                    generated_response_text = generate_response(rag_prompt)
                st.info(generated_response_text)
            else:
                st.write("No hay contexto para generar una respuesta RAG.")
        else:
            st.write("No se encontraron resultados para tu consulta.")
    elif search_button: # Solo mostrar si se apretó el botón pero no hubo resultados
        st.write("No se encontraron resultados para tu consulta.")