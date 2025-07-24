import streamlit as st
from app.retriever import retrieve_by_image, retrieve_by_text
from app.generator import build_context_from_labels, generate_response
import tempfile
from PIL import Image

st.title("Sistema Multimodal + RAG")

mode = st.radio("Modo de búsqueda", ["Imagen", "Texto"])

if mode == "Imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.image(tmp_path, caption="Consulta")
        imgs, descs = retrieve_by_image(tmp_path)
        labels = descs  # en este caso, 'descs' son los labels
        context = build_context_from_labels(labels)
        resp = generate_response(context, "Describe la imagen consultada.")
        st.markdown("### Resultados")
        for i, d in zip(imgs, descs):
            st.image(i, width=150)
            st.write(d)
        st.markdown("### Respuesta generada:")
        st.write(resp)

else:
    query = st.text_input("Escribe tu consulta textual")
    if query:
        imgs, descs = retrieve_by_text(query)
        labels = descs  # en este caso, 'descs' son los labels
        context = build_context_from_labels(labels)
        resp = generate_response(context, query)
        st.markdown("### Resultados")
        for i, d in zip(imgs, descs):
            st.image(i, width=150)
            st.write(d)
        st.markdown("### Respuesta generada:")
        st.write(resp)
