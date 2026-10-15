# RAG con Gemini + Streamlit

Aplicación web para consultar documentos PDF, DOCX y TXT mediante Retrieval-Augmented Generation (RAG) con Google Gemini y ChromaDB.

## Características

- Subida de múltiples archivos PDF, DOCX y TXT desde el navegador.
- Extracción de texto conservando nombre de archivo y página de origen.
- División en fragmentos y generación de embeddings con Google Gemini (`gemini-embedding-001`).
- Base vectorial ligera con NumPy y persistencia en disco (sin servidor externo ni compiladores de C++).
- Chat que responde a partir de los fragmentos recuperados, citando fuentes.
- Detección de "no hay información suficiente".
- Aislamiento por sesión de navegador: los documentos de un usuario no se mezclan con los de otro.
- Eliminación de documentos y limpieza de sesión.
- Diseñado para desplegar en **Streamlit Community Cloud** sin Docker.

## Tecnologías

- Python 3.13
- Streamlit
- google-genai (SDK oficial de Google Gemini)
- NumPy (búsqueda vectorial por similitud de coseno)
- pypdf, python-docx, python-dotenv

## Estructura del proyecto

```
RAG/
├── streamlit_app.py          # Punto de entrada para Streamlit
├── requirements.txt          # Dependencias de Python
├── README.md                 # Este archivo
├── .gitignore                # Archivos ignorados por Git
├── .env.example              # Ejemplo de variables de entorno (sin credenciales)
├── src/
│   ├── __init__.py
│   ├── config.py             # Carga de configuración
│   ├── document_loader.py    # Extracción PDF/DOCX/TXT
│   ├── chunking.py           # División en fragmentos
│   ├── embeddings.py         # Embeddings con Gemini
│   ├── vectorstore.py        # Base vectorial NumPy por sesión
│   ├── rag_chain.py          # Generación de respuestas con fuentes
│   └── session_state.py      # Estado de sesión de Streamlit
└── tests/
    ├── __init__.py
    └── test_basic.py         # Pruebas básicas (sin API key)
```

## Requisitos previos

- Python 3.11 o superior (probado con Python 3.13.1).
- pip.

No se requiere Docker, Node.js ni n8n.

## Instalación local (Windows)

1. Clona o descarga este repositorio en una carpeta.

2. Crea y activa un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instala las dependencias:

```powershell
pip install -r requirements.txt
```

4. Configura la API key de Google Gemini.

Copia `.env.example` a `.env` y edita `.env` con tu clave real:

```powershell
copy .env.example .env
```

Edita `.env` y reemplaza `tu_api_key_de_gemini_aqui` por tu clave de [Google AI Studio](https://aistudio.google.com/app/apikey).

> **Seguridad:** `.env` está incluido en `.gitignore`. Nunca subas credenciales reales al repositorio.

5. Ejecuta la aplicación:

```powershell
streamlit run streamlit_app.py
```

Se abrirá en tu navegador (por defecto `http://localhost:8501`).

## Uso

1. En la barra lateral, selecciona uno o más archivos PDF, DOCX o TXT.
2. Haz clic en **Procesar archivos seleccionados**.
3. Espera a que se confirmen los fragmentos indexados.
4. Escribe tu pregunta en el chat.
5. Lee la respuesta y revisa las fuentes expandidas (nombre de archivo y página).
6. Para eliminar un documento, usa el botón **Eliminar** junto al nombre del archivo.
7. Para borrar todo y comenzar de nuevo, usa **Limpiar sesión**.

## Despliegue en Streamlit Community Cloud

1. Sube este código a un repositorio público de GitHub.
2. Ve a [Streamlit Community Cloud](https://streamlit.io/cloud) y crea una nueva app apuntando al repositorio.
3. Archivo principal: `streamlit_app.py`.
4. En la sección **Secrets** de la app en Streamlit Cloud, agrega:

```toml
GOOGLE_API_KEY = "tu_api_key_de_gemini"
```

5. Despliega la aplicación.

### Notas importantes para Streamlit Cloud

- La base vectorial local (`./vector_store`) es **efímera** en Streamlit Cloud. Los documentos se indexan durante la sesión activa de la app y se pierden cuando el contenedor se reinicia. Los usuarios deben volver a subir sus archivos tras un reinicio.
- Cada sesión de navegador tiene su propio `st.session_state` y sus propios archivos de vectores (`session_<id>.npz` / `.json`), lo que brinda aislamiento práctico entre usuarios dentro de los límites del entorno compartido de Streamlit Cloud.
- No se almacena ninguna credencial en el repositorio.

## Pruebas

Ejecuta las pruebas básicas (no requieren API key):

```powershell
python -m pytest tests/
# o
python -m unittest tests.test_basic
```

## Seguridad

- `GOOGLE_API_KEY` se lee desde variables de entorno o Secrets de Streamlit Cloud.
- `.env` y `.venv` están en `.gitignore`.
- Se valida el tipo y tamaño de los archivos subidos (máximo 10 MB por archivo, configurable).
- No se permite ejecución de código contenido en los documentos subidos.

## Limitaciones conocidas

- PDF escaneados (sin capa de texto) no se pueden procesar sin OCR.
- La base vectorial NumPy almacena los vectores en disco local; en Streamlit Cloud no persiste entre reinicios del contenedor.
- Para aislamiento estricto entre usuarios en producción, se recomienda una base vectorial gestionada por usuario o colecciones con nombres únicos y borrado al finalizar la sesión.