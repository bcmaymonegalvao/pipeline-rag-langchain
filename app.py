import psutil
import streamlit as st
import time
import os
import tempfile
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configuração da página
st.set_page_config(
    page_title="🚀 Pipeline RAG LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #636e72;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .doc-card {
        background: #f8f9fa;
        border-left: 4px solid #6c5ce7;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .response-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
    }    
    
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_system_stats():
    """Retorna estatísticas básicas do sistema para exibição"""
    ram = psutil.virtual_memory()
    faiss_size_mb = 0
    total_vectors = 0
    total_docs = len(st.session_state.docs) if 'docs' in st.session_state else 0

    # Obter tamanho do índice FAISS se existir
    if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
        try:
            faiss_index = st.session_state.vectorstore.index
            faiss_size_mb = faiss_index.ntotal * 4 / (1024*1024)
            total_vectors = faiss_index.ntotal
        except Exception as e:
            logger.warning(f"Erro ao obter tamanho FAISS: {e}")

    return {
        "ram_used": ram.used / (1024**3),
        "ram_total": ram.total / (1024**3),
        "faiss_size_mb": faiss_size_mb,
        "total_vectors": total_vectors,
        "total_docs": total_docs,
    }

def check_system_safety():
    """
    Verifica se os recursos do sistema estão dentro de limites seguros.
    Retorna um dicionário com status de segurança e métricas.
    """
    stats = get_system_stats()
    
    RAM_USAGE_THRESHOLD = 0.85
    FAISS_SIZE_THRESHOLD_GB = 8.0
    
    current_ram_used_gb = stats["ram_used"]
    total_ram_gb = stats["ram_total"]
    ram_usage_pct = current_ram_used_gb / total_ram_gb if total_ram_gb > 0 else 0
    
    faiss_size_gb = stats["faiss_size_mb"] / 1024
    
    return {
        "current_ram_used_gb": current_ram_used_gb,
        "total_ram_gb": total_ram_gb,
        "ram_usage_pct": ram_usage_pct,
        "faiss_size_gb": faiss_size_gb,
        "current_ram_safe": ram_usage_pct < RAM_USAGE_THRESHOLD,
        "faiss_size_safe": faiss_size_gb < FAISS_SIZE_THRESHOLD_GB,
        "overall_safe": ram_usage_pct < RAM_USAGE_THRESHOLD and faiss_size_gb < FAISS_SIZE_THRESHOLD_GB
    }

def get_theoretical_limits():
    """Retorna limites teóricos baseados em configurações e hardware"""
    return {
        "max_vectors_ram": 1000000,
        "max_docs_estimate": 20000,
        "max_faiss_size_gb": 10.0,
        "max_context_tokens": 2048,
    }

def get_default_docs():
    """Retorna os documentos padrão"""
    return [
        "O churn é o cancelamento ou abandono de clientes em um serviço ou produto. É uma métrica crucial para avaliar a retenção e satisfação do cliente.",
        "NPS, ou Net Promoter Score, mede a lealdade dos clientes através da pergunta: 'Você recomendaria nossa empresa a um amigo?' Varia de -100 a +100.",
        "LangChain é uma biblioteca para construir aplicações que usam modelos de linguagem large (LLMs) integrados a outras fontes de dados e ferramentas.",
        "RAG, Retrieval-Augmented Generation, conecta modelos de linguagem a bases de conhecimento através de embeddings e mecanismos de busca para melhorar respostas.",
        "Embeddings representam texto em vetores numéricos que capturam significado semântico, possibilitando busca eficiente por similaridade.",
        "O pipeline básico de RAG inclui: criação de embeddings, uso do retriever para buscar documentos relevantes e geração da resposta pelo LLM.",
        "Machine Learning é um subcampo da inteligência artificial que permite que sistemas aprendam e melhorem automaticamente a partir da experiência.",
        "Deep Learning utiliza redes neurais artificiais com múltiplas camadas para modelar e entender dados complexos como imagens, texto e áudio.",
        "Natural Language Processing (NLP) é a área da IA focada na interação entre computadores e linguagem humana natural.",
        "Business Intelligence (BI) envolve estratégias e tecnologias para análise de informações de negócios e suporte à tomada de decisão."
    ]

def process_pdf(uploaded_file):
    """Processa um arquivo PDF e extrai o texto"""
    tmp_file_path = None
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        docs = text_splitter.split_documents(pages)
        return docs
        
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}")
        raise e
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Erro ao deletar arquivo temporário: {e}")

def process_pdf_safely(uploaded_file, max_chunks_per_file=200):
    """
    Processa um arquivo PDF com segurança limitando número de chunks.
    Retorna uma tupla (lista de docs, lista de warnings).
    """
    warnings = []
    docs = []
    try:
        loaded_docs = process_pdf(uploaded_file)
        if len(loaded_docs) > max_chunks_per_file:
            warnings.append(f"⚠️ Arquivo muito grande, cortado para {max_chunks_per_file} chunks")
            docs = loaded_docs[:max_chunks_per_file]
        else:
            docs = loaded_docs
    except Exception as e:
        warnings.append(f"⚠️ Erro ao processar PDF: {str(e)}")
        docs = []
    return docs, warnings

def save_custom_docs(docs_list):
    """Salva a lista de documentos customizados"""
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/custom_docs.pkl", "wb") as f:
            pickle.dump(docs_list, f)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar documentos: {e}")
        return False

def load_custom_docs():
    """Carrega a lista de documentos customizados"""
    try:
        if os.path.exists("data/custom_docs.pkl"):
            with open("data/custom_docs.pkl", "rb") as f:
                return pickle.load(f)
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar documentos: {e}")
        return None

def format_response(response_data: Dict[str, Any]) -> tuple:
    """Formata a resposta do RAG para exibição"""
    if isinstance(response_data, dict):
        answer = response_data.get("result", "Resposta não encontrada")
        source_docs = response_data.get("source_documents", [])
    else:
        answer = str(response_data)
        source_docs = []
    
    answer = answer.strip()
    if not answer or answer.lower() in ["", "none", "null"]:
        answer = "Desculpe, não consegui gerar uma resposta adequada para sua pergunta."
    
    return answer, source_docs

@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline(custom_docs=None):
    """Inicializa o pipeline RAG com cache para otimizar performance"""
    try:
        with st.spinner("🔧 Inicializando pipeline RAG..."):
            from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
            from langchain_community.llms import HuggingFacePipeline
            
            docs = custom_docs if custom_docs else get_default_docs()
            
            st.info("📊 Carregando modelo de embeddings MiniLM...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            st.info("🗃️ Construindo índice vetorial FAISS...")
            vectorstore = FAISS.from_texts(docs, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            st.info("🤖 Carregando modelo FLAN-T5...")
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            gen_pipeline = hf_pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            llm = HuggingFacePipeline(pipeline=gen_pipeline)
            
            st.info("🔗 Configurando chain RAG...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                verbose=False
            )
            
            st.success("✅ Pipeline RAG inicializado com sucesso!")
            return qa_chain, vectorstore, docs, retriever, embeddings
            
    except Exception as e:
        st.error(f"❌ Erro ao inicializar pipeline: {str(e)}")
        logger.error(f"Erro na inicialização: {e}")
        return None, None, None, None, None

def show_sidebar():
    """Mostra a sidebar com informações do sistema"""
    with st.sidebar:
        st.header("📍 Navegação")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Chat", use_container_width=True):
                st.session_state.current_page = "chat"
        with col2:
            if st.button("📚 Documentos", use_container_width=True):
                st.session_state.current_page = "manage"
        
        st.markdown("---")
        
        st.header("ℹ️ Sobre o Sistema")
        st.markdown("""
        **🎯 Tecnologias:**
        - 🧠 **LLM**: FLAN-T5 (Google)
        - 🔍 **Embeddings**: MiniLM (HuggingFace) 
        - 📊 **Vector Store**: FAISS
        - 🔗 **Framework**: LangChain
        
        **⚡ Características:**
        - ✅ 100% Gratuito (sem API keys)
        - 🖥️ Roda localmente
        - 🚀 Interface moderna
        - 📚 Base de conhecimento expansível
        - 📄 Upload de PDFs
        """)
        
        st.header("📋 Como usar")
        if st.session_state.current_page == "chat":
            st.markdown("""
            1. Digite sua pergunta
            2. Clique em "Buscar Resposta"
            3. Veja a resposta e documentos
            """)
        else:
            st.markdown("""
            1. Faça upload de PDFs
            2. Processe os documentos
            3. Teste a base atualizada
            """)
        
        st.header("📊 Sistema")
        if 'qa_chain' in st.session_state:
            st.success("🟢 Pipeline Ativo")
            st.info(f"📄 Docs: {len(st.session_state.docs)}")
        else:
            st.warning("🟡 Inicializando...")

def chat_page():
    """Página principal do chat"""
    st.markdown('<div class="main-header">🚀 Pipeline RAG LangChain</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Retrieval-Augmented Generation com HuggingFace & FAISS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Faça sua pergunta")
        
        query = st.text_area(
            "Digite sua pergunta aqui:",
            placeholder="Ex: O que é RAG? Como funciona machine learning?",
            height=100,
            key="chat_input"
        )
        
        st.markdown("**💡 Perguntas de exemplo:**")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        example_queries = [
            ("🔄 O que é churn?", "O que significa churn em análise de clientes?"),
            ("🤖 Como funciona RAG?", "Como funciona o pipeline de RAG?"),
            ("📚 O que é LangChain?", "Explique o que é LangChain e para que serve.")
        ]
        
        for i, (btn_text, query_text) in enumerate(example_queries):
            with [col_btn1, col_btn2, col_btn3][i]:
                if st.button(btn_text, use_container_width=True):
                    st.session_state.chat_input = query_text
                    st.rerun()
        
        search_button = st.button("🔍 Buscar Resposta", type="primary", use_container_width=True)
        
        if search_button and query.strip():
            with st.spinner("🤔 Processando sua pergunta..."):
                try:
                    start_time = time.time()
                    response = st.session_state.qa_chain.invoke({"query": query})
                    response_time = time.time() - start_time
                    
                    answer, source_docs = format_response(response)
                    
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": answer,
                        "time": response_time,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    
                    st.markdown('<div class="response-card">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 Resposta:**\n\n{answer}")
                    st.markdown(f"⏱️ *Tempo de resposta: {response_time:.2f}s*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if source_docs:
                        st.markdown("**📄 Documentos utilizados:**")
                        for i, doc in enumerate(source_docs[:2], 1):
                            with st.expander(f"📋 Fonte {i}"):
                                st.markdown(f'<div class="doc-card">{doc.page_content}</div>', 
                                          unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar pergunta: {str(e)}")
                    logger.error(f"Erro no processamento: {e}")
        
        elif search_button:
            st.warning("⚠️ Por favor, digite uma pergunta antes de buscar!")
    
    with col2:
        st.header("📈 Estatísticas")
        
        if 'query_history' in st.session_state and st.session_state.query_history:
            total_queries = len(st.session_state.query_history)
            avg_time = sum(q["time"] for q in st.session_state.query_history) / total_queries
            last_query_time = st.session_state.query_history[-1]["time"]
            
            st.metric("🔢 Total de Perguntas", total_queries)
            st.metric("⏱️ Tempo Médio", f"{avg_time:.2f}s")
            st.metric("🕐 Última Consulta", f"{last_query_time:.2f}s")
        else:
            st.metric("🔢 Total de Perguntas", "0")
            st.metric("⏱️ Tempo Médio", "-")
            st.metric("🕐 Última Consulta", "-")
        
        st.header("📚 Base de Conhecimento")
        st.info(f"📄 {len(st.session_state.docs)} documentos indexados")
        
        with st.expander("👁️ Ver documentos"):
            for i, doc in enumerate(st.session_state.docs[:5], 1):
                st.markdown(f"**{i}.** {doc[:100]}...")
            if len(st.session_state.docs) > 5:
                st.markdown(f"*... e mais {len(st.session_state.docs) - 5} documentos*")

    if 'query_history' in st.session_state and st.session_state.query_history:
        st.header("📋 Histórico de Perguntas")
        recent_history = st.session_state.query_history[-5:]
        
        for item in reversed(recent_history):
            with st.expander(f"🕐 {item['timestamp']} - {item['query'][:50]}..."):
                st.markdown(f"**Pergunta:** {item['query']}")
                st.markdown(f"**Resposta:** {item['answer']}")
                st.markdown(f"**Tempo:** {item['time']:.2f}s")

def manage_documents_page():
    """Página de gerenciamento de documentos"""
    st.markdown('<div class="main-header">📚 Gerenciar Base de Conhecimento</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adicione documentos PDF para expandir o conhecimento da IA</div>', unsafe_allow_html=True)
    
    st.header("📤 Upload de Documentos")
    st.markdown("**📄 Adicione arquivos PDF para expandir a base de conhecimento**")
    
    uploaded_files = st.file_uploader(
        "Selecione arquivos PDF:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Você pode selecionar múltiplos arquivos PDF para upload simultâneo."
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} arquivo(s) selecionado(s)**")
        
        for file in uploaded_files:
            st.markdown(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🚀 Processar e Adicionar à Base", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                all_new_docs = []
                new_texts = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"📄 Processando {uploaded_file.name}...")
                    progress_bar.progress((i + 0.5) / len(uploaded_files))
                    
                    try:
                        docs, warnings = process_pdf_safely(uploaded_file, max_chunks_per_file=200)
                        all_new_docs.extend(docs)
                        
                        for warning in warnings:
                            if "⚠️" in warning:
                                st.warning(f"{uploaded_file.name}: {warning}")
                        
                        for doc in docs:
                            new_texts.append(doc.page_content)
                        
                        st.markdown(f'<div class="success-card">✅ {uploaded_file.name}: {len(docs)} chunks extraídos</div>', 
                                  unsafe_allow_html=True)
                        
                        current_safety = check_system_safety()
                        if not current_safety['current_ram_safe']:
                            st.error("⛔ Recursos críticos! Interrompendo processamento.")
                            break
                        
                    except Exception as e:
                        st.markdown(f'<div class="warning-card">❌ Erro ao processar {uploaded_file.name}: {e}</div>', 
                                  unsafe_allow_html=True)
                        continue
                
                if new_texts:
                    status_text.text("🔄 Atualizando base de conhecimento...")
                    progress_bar.progress(0.9)
                    
                    updated_docs = st.session_state.docs + new_texts
                    
                    if save_custom_docs(updated_docs):
                        st.cache_resource.clear()
                        
                        qa_chain, vectorstore, docs, retriever, embeddings = initialize_rag_pipeline(updated_docs)
                        
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.session_state.vectorstore = vectorstore
                            st.session_state.docs = docs
                            st.session_state.retriever = retriever
                            st.session_state.embeddings = embeddings
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Processamento concluído!")
                            
                            st.markdown(f'<div class="success-card">🎉 <strong>Sucesso!</strong> {len(new_texts)} novos documentos adicionados à base de conhecimento.</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.error("❌ Erro ao reinicializar pipeline com novos documentos.")
                    else:
                        st.error("❌ Erro ao salvar documentos atualizados.")
                else:
                    st.warning("⚠️ Nenhum documento foi processado com sucesso.")
                    
            except Exception as e:
                st.error(f"❌ Erro durante o processamento: {e}")
                status_text.text("❌ Erro no processamento")
                progress_bar.progress(0)
    
    st.markdown("---")
    st.header("📊 Informações da Base Atual")
    
    stats = get_system_stats()
    limits = get_theoretical_limits()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**📄 Documentos**\n{stats['total_docs']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**🔢 Vetores**\n{stats['total_vectors']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**💾 FAISS**\n{stats['faiss_size_mb']:.1f} MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.header("🔧 Recursos do Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💾 Memória RAM:**")
        ram_usage = (stats['ram_used'] / stats['ram_total']) * 100
        st.progress(ram_usage / 100)
        st.markdown(f"**{stats['ram_used']:.1f} GB** / {stats['ram_total']:.1f} GB ({ram_usage:.1f}%)")
        
        if ram_usage > 80:
            st.warning("⚠️ Uso alto de memória!")
        elif ram_usage > 60:
            st.info("ℹ️ Uso moderado de memória")
        else:
            st.success("✅ Uso normal de memória")
    
    with col2:
        st.markdown("**📈 Limites Teóricos:**")
        st.info(f"🔢 **Max. Vetores:** {limits['max_vectors_ram']:,}")
        st.info(f"📄 **Max. Documentos:** {limits['max_docs_estimate']:,}")
        st.info(f"💾 **Max. FAISS:** {limits['max_faiss_size_gb']:.1f} GB")
        st.info(f"📝 **Context Window:** {limits['max_context_tokens']} tokens")
    
    st.header("🔍 Testar Base de Conhecimento")
    test_query = st.text_input(
        "Digite uma consulta para testar:",
        placeholder="Ex: machine learning, churn, RAG",
        key="test_query"
    )
    
    if st.button("🔍 Buscar documentos similares") and test_query:
        with st.spinner("🔍 Buscando documentos similares..."):
            try:
                retriever = st.session_state.retriever
                docs = retriever.get_relevant_documents(test_query)
                
                if docs:
                    st.markdown("**📄 Documentos encontrados:**")
                    for i, doc in enumerate(docs[:3], 1):
                        with st.expander(f"📋 Documento {i}", expanded=i==1):
                            st.markdown(f'<div class="doc-card">{doc.page_content}</div>', 
                                      unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Nenhum documento relevante encontrado.")
                    
            except Exception as e:
                st.error(f"❌ Erro na busca: {e}")
    
    st.markdown("---")
    st.header("🛠️ Gerenciamento Avançado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Recarregar Pipeline", use_container_width=True):
            with st.spinner("🔄 Recarregando pipeline..."):
                st.cache_resource.clear()
                st.rerun()
    
    with col2:
        if st.button("⚠️ Resetar Base", use_container_width=True):
            if st.checkbox("✅ Confirmo que quero resetar para documentos padrão"):
                try:
                    if os.path.exists("data/custom_docs.pkl"):
                        os.remove("data/custom_docs.pkl")
                    
                    st.cache_resource.clear()
                    st.success("✅ Base resetada! Recarregue a página para aplicar.")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao resetar: {e}")

def main():
    """Função principal da aplicação"""
    # Inicializar estado da sessão
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Mostrar sidebar
    show_sidebar()
    
    # Inicializar pipeline (apenas uma vez)
    if 'qa_chain' not in st.session_state:
        custom_docs = load_custom_docs()
        qa_chain, vectorstore, docs, retriever, embeddings = initialize_rag_pipeline(custom_docs)
        
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.vectorstore = vectorstore
            st.session_state.docs = docs
            st.session_state.retriever = retriever
            st.session_state.embeddings = embeddings
        else:
            st.stop()
    
    # Renderizar página baseada na seleção
    if st.session_state.current_page == "chat":
        chat_page()
    else:
        manage_documents_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #636e72;'>
        🚀 Desenvolvido com Streamlit | 🤖 Powered by HuggingFace & LangChain
        <br>
        💡 Sistema RAG 100% gratuito e local | 📄 Agora com suporte a upload de PDFs
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
