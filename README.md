# Sistema de Recuperación Multimodal de Información (SRI Multimodal RAG)

Este proyecto implementa un Sistema de Recuperación Multimodal de Información (SRI) que combina la búsqueda por texto e imagen con la Generación Aumentada por Recuperación (RAG). El objetivo es permitir a los usuarios consultar un corpus de imágenes y texto, recuperar contenido relevante en base a la entrada multimodal, y generar una respuesta narrativa o explicativa utilizando un modelo de lenguaje grande (LLM).

## Características

* **Codificación e Indexación Multimodal:**
    * Utiliza el modelo CLIP para codificar imágenes y textos en un espacio vectorial común.
    * Construye un índice vectorial FAISS para una búsqueda eficiente de embeddings.
* **Consulta Multimodal Flexible:**
    * Permite al usuario subir una imagen para encontrar imágenes y descripciones similares.
    * Permite al usuario ingresar una consulta textual para encontrar imágenes y descripciones relevantes.
* **Generación Aumentada por Recuperación (RAG):**
    * Concatena las descripciones recuperadas para formar un contexto enriquecido.
    * Utiliza un modelo generativo (e.g., GPT de OpenAI) para producir una respuesta coherente y basada en el contexto recuperado.
* **Interfaz de Usuario Amigable:**
    * Ofrece una interfaz web intuitiva construida con Streamlit para interactuar con el sistema.
    * Visualización clara de los resultados de recuperación (imágenes y descripciones).
    * Visualización de la respuesta generada por el LLM.

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
├── main.py                            # Archivo principal para pruebas
├── requirements.txt                   # Dependencias del proyecto (librerías necesarias)
├── config.json                        # Archivo JSON para parámetros de configuración adicionales (por ejemplo, API keys, etc.)
├── README.md                          # Documentación general del proyecto
└── .gitignore                         # Archivos a ignorar para git (datasets grandes, archivos temporales, etc.)
```

## Configuración y Ejecución

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local.

### 1. Requisitos Previos

* **Python 3.8+**
* **Git**

### 2. Clonar el Repositorio

```bash
git clone [https://github.com/nMishelRamirez/SRI_Multimodal.git](https://github.com/nMishelRamirez/SRI_Multimodal.git)
cd SRI_Multimodal
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Recursos de NLTK

Algunas funciones de preprocesamiento (en src/preprocessing.py) requieren datos de NLTK. Ejecuta el siguiente comando una vez para descargarlos si no cuentas con ellos:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Dataset

Este proyecto está configurado para usar el dataset Flickr8k.

**1.  Descargar el dataset**
Descarga el dataset Flickr8k. Necesitarás tanto las imágenes como el archivo de anotaciones (tokens).

    - Imágenes: Flickr8k_Dataset
    - Anotaciones: Flickr8k_text

Puedes encontrarlos en sitios como Kaggle o la página oficial del dataset.

**2. Organizar el Directorio dataset**

Crea un directorio data en la raíz del proyecto y move los archivos de la carpeta Flicker8k_Dataset y Flickr8k_text dentro de él.

### 6. Configurar la Clave API de OpenAI

Para la funcionalidad RAG con modelos de OpenAI, necesitas una clave API.

1. Crea un archivo llamado config.json en el directorio raíz del proyecto (SRI_Multimodal/).

2. Añade tu clave API de OpenAI en este archivo.

### 7. Ejecutar el Proyecto

Ejecuta el archivo build_index.py para que genera los embeddings y construye el índice de búsqueda. Puede tardar un tiempo dependiendo del tamaño del dataset y tu hardware.

```bash
python -m src.build_index
```

Luego, ejecuta el archivo main.py para pruebas.

```bash
python -m main
```

### 8. Ejecutar la Interfaz Web

Puedes ejecutar la interfaz web para consultar imágenes y textos utilizando el archivo interface.py.

```bash
streamlit run web/interface.py
```

## Integrantes

- Mishel Ramirez
- Danna Zaldumbide
