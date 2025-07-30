# SRI_Multimodal


## Estructura del proyecto

```
SRI_Multimodal/                        # Directorio principal del proyecto
│
├── data/                              # Directorio de datos
│   ├── Flicker8k_Dataset/             # Contiene las imágenes del corpus Flickr8k
│   │   └── imagenes/                  # Contiene las imágenes del corpus
│   └── Flickr8k_text/                 # Contiene los archivos de texto (Flickr8k.token.txt)
│       └── Flickr8k.token.txt         # Archivo de texto con las descripciones de las imágenes
├── src/                               # Código fuente del proyecto
│   ├── config.py                      # Archivo para configurar parámetros globales (paths, API keys, etc.)
│   ├── data_loader.py                 # Código para cargar y procesar los datos
│   ├── preprocessing.py               # Código para preprocesar datos 
│   ├── embedding.py                   # Código para generar embeddings
│   ├── indexer.py                     # Código para construir el índice vectorial (FAISS)
│   ├── retriever.py                   # Funciones para recuperar descripciones/imágenes
│   ├── generator.py                   # Código para la generación de respuestas (Mini-RAG usando GPT u otro modelo generativo)
│   └── build_index.py                 # Script para construir el índice
│
├── web/                               # Directorio para la interfaz web
│   └── interface.py                   # Aplicación web principal Streamlit
│
├── __init__.py                        # Archivo para importar el proyecto
├── requirements.txt                   # Dependencias del proyecto (librerías necesarias)
├── config.json                        # Archivo JSON para parámetros de configuración adicionales (por ejemplo, API keys, etc.)
├── README.md                          # Documentación general del proyecto
└── .gitignore                         # Archivos a ignorar para git (datasets grandes, archivos temporales, etc.)
```