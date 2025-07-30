# SRI_Multimodal/web/interface.py
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
    st.write("Cargando base de datos FAISS...")
    faiss_db_instance = FaissVectorDB.load_index()
    if faiss_db_instance is None:
        st.error("Error al cargar el índice FAISS. Por favor, asegúrate de haber ejecutado 'python src/build_index.py' primero.")
        st.stop()
        
    return faiss_db_instance

faiss_db = load_all_resources()

st.set_page_config(layout="wide", page_title="SRI Multimodal RAG")
st.title("Sistema de Recuperación Multimodal de Información (RAG)")

st.sidebar.header("Opciones")
query_type = st.sidebar.radio("Selecciona el tipo de consulta:", ("Imagen", "Texto"))

st.header("Realizar Búsqueda")

query_input_text = ""
uploaded_file = None
retrieved_results = []
generated_response_text = "Esperando consulta..."

if query_type == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen:", type=["jpg", "jpeg", "png"])
    if st.button("Buscar por Imagen", key="search_image_button") and uploaded_file is not None:
        query_input_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_input_image, caption='Imagen de consulta', width=250)
        
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Procesando imagen y buscando resultados..."):
            try:
                distances, indices = retrieve_by_image(temp_image_path, faiss_db, TOP_K_RETRIEVAL)
                os.remove(temp_image_path)
                
                if indices is not None and len(indices[0]) > 0:
                    retrieved_results = [faiss_db.image_info[idx] for idx in indices[0] if idx != -1 and idx < len(faiss_db.image_info)]
                else:
                    st.info("No se encontraron resultados relevantes para la imagen.")
            except Exception as e:
                st.error(f"Error al procesar la imagen de consulta o buscar: {e}")
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    elif st.button("Buscar por Imagen", key="search_image_no_file_button") and uploaded_file is None:
        st.warning("Por favor, sube una imagen para la búsqueda.")

elif query_type == "Texto":
    query_input_text = st.text_input("Ingresa tu consulta de texto:")
    if st.button("Buscar por Texto", key="search_text_button") and query_input_text:
        with st.spinner("Procesando texto y buscando resultados..."):
            try:
                distances, indices = retrieve_by_text(query_input_text, faiss_db, TOP_K_RETRIEVAL)
                if indices is not None and len(indices[0]) > 0:
                    retrieved_results = [faiss_db.image_info[idx] for idx in indices[0] if idx != -1 and idx < len(faiss_db.image_info)]
                else:
                    st.info("No se encontraron resultados relevantes para el texto.")
            except Exception as e:
                st.error(f"Error al procesar el texto de consulta o buscar: {e}")
    elif st.button("Buscar por Texto", key="search_text_no_input_button"):
        st.warning("Por favor, ingresa una consulta de texto.")

st.header("Resultados Recuperados")
if retrieved_results:
    num_cols = len(retrieved_results)
    cols = st.columns(num_cols if num_cols > 0 else 1)
    
    context_for_rag = []
    for i, res in enumerate(retrieved_results):
        with cols[i]:
            try:
                st.image(Image.open(res['image_path']).convert("RGB"), caption=f"ID: {os.path.basename(res['file_name'])}", width=150)
                st.markdown(f"**Descripción:** {res['caption']}")
                context_for_rag.append(res['caption'])
            except Exception as e:
                st.error(f"Error al cargar imagen {res['image_path']}: {e}")
                st.markdown(f"**Descripción:** {res['caption']}")
                context_for_rag.append(res['caption'])
    
    st.header("Respuesta Generada (RAG)")
    if context_for_rag:
        final_query_for_llm = query_input_text if query_input_text else "Describe las imágenes recuperadas."
        
        with st.spinner(f"Generando respuesta narrativa/explicativa con {GENERATIVE_MODEL_NAME}..."):
            rag_prompt = build_prompt(final_query_for_llm, context_for_rag)
            generated_response_text = generate_response(rag_prompt)
        st.info(generated_response_text)
    else:
        st.info("No se encontraron descripciones relevantes para generar una respuesta.")
else:
    st.info("No se encontraron resultados relevantes. Intenta una consulta diferente.")
    st.header("Respuesta Generada (RAG)")
    st.info(generated_response_text)