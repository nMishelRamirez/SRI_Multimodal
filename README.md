# SRI_Multimodal


## Estructura del proyecto

```
SRI_Multimodal/                        # Directorio principal del proyecto
│
├── data/                              # Directorio de datos
│   ├── raw/                           # Datos crudos
│   │   ├── images/                    # Subcarpeta con imágenes
│   │   ├── captions.csv               # Corpus 1: [label, caption]
│   │   └── concept_descriptions.csv   # Corpus 2: [label, description]
│   │
│   └── processed/                     # Datos generados automáticamente
│       ├── clip_index.faiss           # Índice vectorial de CLIP
│       ├── image_paths.npy            # Rutas de imágenes
│       └── descriptions.npy           # Descripciones
│
├── notebooks/                         # Jupyter Notebooks para exploración de datos, pruebas y análisis
│   └── exploracion_datos.ipynb        # Notebook con exploración y visualización de datasets
│
├── src/                               # Código fuente del proyecto
│   ├── __init__.py                    # Hace que src sea un paquete de Python
│   ├── config.py                      # Archivo para configurar parámetros globales (paths, API keys, etc.)
│   ├── preprocessing.py               # Código para preprocesar datos (imagenes y textos)
│   ├── encoding.py                    # Código para codificar imágenes y textos usando CLIP u otros encoders
│   ├── indexing.py                    # Código para construir el índice vectorial (FAISS)
│   ├── retrieval.py                   # Funciones para recuperar descripciones/imágenes
│   ├── generation.py                  # Código para la generación de respuestas (Mini-RAG usando GPT u otro modelo generativo)
│   └── utils.py                       # Funciones auxiliares comunes (por ejemplo, para visualización)
│
├── web/                               # Directorio para la interfaz web
│   └── interface.py                   # Aplicación web principal Streamlit
│
├── scripts/                           # Scripts auxiliares    
│   ├── build_index.py                 # Script para construir el índice
│   └── run_app.py                     # Script para ejecutar la interfaz web
│
├── requirements.txt                   # Dependencias del proyecto (librerías necesarias)
├── config.json                        # Archivo JSON para parámetros de configuración adicionales (por ejemplo, API keys, etc.)
├── README.md                          # Documentación general del proyecto
└── .gitignore                         # Archivos a ignorar para git (datasets grandes, archivos temporales, etc.)
```