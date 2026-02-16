import google.generativeai as genai
import os
from dotenv import load_dotenv


def read_key_from_streamlit_secrets(key_name):
    """Read a top-level key from .streamlit/secrets.toml."""
    secrets_path = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_path):
        return None

    try:
        with open(secrets_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key_name:
                    return v.strip().strip('"').strip("'")
    except Exception:
        return None

    return None


load_dotenv()
api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or read_key_from_streamlit_secrets("GEMINI_API_KEY")
)

if not api_key:
    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY/GOOGLE_API_KEY or add GEMINI_API_KEY to .streamlit/secrets.toml"
    )

genai.configure(api_key=api_key)

# Show embed-capable models
for m in genai.list_models():
    methods = getattr(m, "supported_generation_methods", []) or []
    if "embedContent" in methods or "batchEmbedContents" in methods:
        print(m.name, methods)

# Actual embedding call
resp = genai.embed_content(
    model="models/embedding-001",
    content="test embedding",
    task_type="retrieval_document",
)
print("Embedding length:", len(resp["embedding"]))