# SRI_Multimodal

SRI_Multimodal/
│
├── app/                     # Código fuente principal
│   ├── __init__.py
│   ├── config.py            # Configuración (paths, API keys, etc.)
│   ├── encoder.py           # Encoders CLIP para texto e imagen
│   ├── indexer.py           # Construcción y carga de índice FAISS
│   ├── retriever.py         # Funciones para recuperar descripciones/imágenes
│   ├── generator.py         # Lógica del modelo generativo (GPT u otro)
│   └── interface.py         # Interfaz web (Streamlit)
│
├── data/                    # Datos
│   ├── images/              # Imágenes del corpus
│   ├── captions.csv         # Corpus 1: [filename, label]
│   ├── concept_descriptions.csv  # Corpus 2: [label, full_description]
│   ├── clip_index.faiss     # Índice FAISS (se genera)
│   ├── image_paths.npy      # Rutas de imágenes (se genera)
│   └── descriptions.npy     # Descripciones de imágenes (se genera)
│
├── notebooks/               # (opcional) Notebooks para pruebas
│
├── scripts/                 # Scripts ejecutables
│   ├── build_index.py       # Script para construir el índice
│   └── run_app.py           # Ejecuta la interfaz Streamlit
│
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Instrucciones y explicación del proyecto
