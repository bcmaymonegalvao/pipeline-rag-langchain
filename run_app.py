#!/usr/bin/env python3
"""
Script para inicializar a aplicação RAG Streamlit
Inclui verificação de dependências e configurações otimizadas
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        print(f"Versão atual: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version.split()[0]} detectado")

def install_requirements():
    """Instala as dependências necessárias"""
    try:
        print("📦 Verificando dependências...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("✅ Dependências instaladas/verificadas")
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        print("💡 Tente executar manualmente: pip install -r requirements.txt")
        sys.exit(1)

def set_environment():
    """Configura variáveis de ambiente para otimização"""
    
    # Configurações para otimizar performance em CPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    
    # Configurações do HuggingFace
    os.environ["TRANSFORMERS_CACHE"] = "./models_cache"
    os.environ["HF_HOME"] = "./models_cache"
    
    print("⚙️ Variáveis de ambiente configuradas")

def run_streamlit():
    """Executa a aplicação Streamlit"""
    try:
        print("🚀 Iniciando aplicação Streamlit...")
        print("🌐 A aplicação será aberta em: http://localhost:8501")
        print("⏹️ Para parar, pressione Ctrl+C\n")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Aplicação finalizada pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar aplicação: {e}")

def main():
    """Função principal"""
    print("🚀 Inicializando Pipeline RAG LangChain")
    print("=" * 50)
    
    # Verificações e configurações
    check_python_version()
    set_environment()
    install_requirements()
    
    print("=" * 50)
    
    # Verificar se o arquivo principal existe
    if not os.path.exists("app.py"):
        print("❌ Arquivo app.py não encontrado!")
        print("💡 Certifique-se de que app.py está no diretório atual")
        sys.exit(1)
    
    # Executar aplicação
    run_streamlit()

if __name__ == "__main__":
    main()