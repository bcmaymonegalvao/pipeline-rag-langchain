"""Entry point para Streamlit Cloud.

Este arquivo importa e executa a aplicação principal de src/app/app.py
"""
import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importa e executa o app principal
from app import app  # noqa: E402
