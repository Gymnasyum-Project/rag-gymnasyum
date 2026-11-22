import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("groq>=0.11.0")
    .env({"HF_HOME": "/root/hf_cache"})
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=["venv", ".venv", "__pycache__", ".git", "chroma_db", "*.pyc"],
    )
)

app = modal.App("gymnasyum-rag-service")
cache_volume = modal.Volume.from_name("gymnasyum-rag-cache", create_if_missing=True)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=300,
    scaledown_window=120,
    volumes={"/root/hf_cache": cache_volume},
    secrets=[modal.Secret.from_dotenv()],
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    os.chdir("/root/app")
    sys.path.append("/root/app")

    from app import app as web_app

    return web_app