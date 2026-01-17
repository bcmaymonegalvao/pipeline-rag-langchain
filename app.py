"""
MIGUEL — Modelo Interativo Generativo de Linguagem para Uso Educacional Livre
----------------------------------------------------------------------------

Aplicativo didático (Streamlit) que demonstra um pipeline RAG (Retrieval-Augmented Generation)
com componentes 100% locais e gratuitos:
- Representações vetoriais (MiniLM) + FAISS (busca por similaridade)
- LLM (FLAN-T5) para geração de resposta
- Upload de PDFs para ampliar a base de conhecimento

Objetivo: permitir que estudantes e professores aprendam IA “por dentro”, personalizando e
observando as etapas do pipeline, com uma interface clara e documentação acessível.

Notas:
- Tema: paleta clara (principal) com acentos escuros (auxiliares).
- Evita TypeError ao formatar métricas quando o histórico está vazio.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import streamlit as st


# =============================================================================
# Configuração geral do app
# =============================================================================

APP_NAME = "Modelo Interativo Generativo de Linguagem para Uso Educacional Livre"
APP_SHORT = "MIGUEL"
APP_SUBTITLE = "Aplicativo didático (RAG) com LLM local, representações vetoriais e FAISS — sem API keys"

st.set_page_config(
    page_title=f"{APP_SHORT} — {APP_NAME}",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Estilo (CSS) + utilidades de UI
# =============================================================================

def inject_minimal_css() -> None:
    """Tema claro (principal) com acentos escuros (auxiliares) e alto contraste no título."""
    st.markdown(
        """
        <style>
          :root{
            /* Base clara */
            --bg: #f7f9fc;
            --panel: #ffffff;
            --card: #ffffff;

            /* Texto */
            --text: #0b1220;
            --muted: #475569;

            /* Linhas/bordas */
            --line: #d6deea;

            /* Acento (auxiliar escuro) */
            --accent: #1e3a8a;       /* azul escuro */
            --accentSoft: #e8f0ff;   /* azul bem claro */

            /* Estados */
            --ok: #15803d;
            --warn: #b45309;
            --err: #b91c1c;
          }

          /* Fundo geral */
          .stApp {
            background: var(--bg);
            color: var(--text);
          }

          /* Sidebar clara */
          section[data-testid="stSidebar"]{
            background: var(--panel);
            border-right: 1px solid var(--line);
          }

          /* Título com contraste alto */
          .miguel-title {
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            text-align: left;
            margin: 0.25rem 0 0.25rem 0;
            color: var(--text);
          }

          .miguel-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin: 0 0 1.15rem 0;
          }

          /* Cards */
          .miguel-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0;
            box-shadow: 0 1px 0 rgba(2, 6, 23, 0.03);
          }

          .miguel-card h3, .miguel-card h4 {
            margin: 0 0 0.35rem 0;
            color: var(--text);
          }

          .miguel-card p {
            margin: 0.2rem 0 0 0;
            color: var(--muted);
          }

          /* Pills */
          .miguel-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--accentSoft);
            color: var(--accent);
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 0.35rem;
          }

          /* Status */
          .state-ok    { color: var(--ok); }
          .state-warn  { color: var(--warn); }
          .state-err   { color: var(--err); }

          /* Links */
          a { color: var(--accent); }

          /* Botões */
          .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--line);
          }

          /* Espaçamento superior */
          .block-container { padding-top: 1.0rem; }
        </style>
        """,
        unsafe_allow_html=True,
        /* --- Evitar que o header do Streamlit cubra o conteúdo --- */
        header[data-testid="stHeader"]{
          background: transparent;   /* não pinta por cima do título */
        }
        
        /* Empurra a área principal para baixo, evitando corte do título */
        div[data-testid="stAppViewContainer"] > .main {
          padding-top: 4.25rem;
        }
        
        /* Mantém o espaçamento interno do container (conteúdo) */
        .block-container {
          padding-top: 0.75rem;
        }
        
        /* Em telas menores, o header pode ser maior */
        @media (max-width: 768px){
        div[data-testid="stAppViewContainer"] > .main {
            padding-top: 5.0rem;
          }
        }
    )


def render_header() -> None:
    """Renderiza cabeçalho do app (sem emoji de foguete)."""
    st.markdown(f"<div class='miguel-title'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='miguel-subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #d6deea;margin:0.6rem 0 1.0rem 0;'>",
        unsafe_allow_html=True,
    )


def card(title: str, body_md: str, pills: Optional[List[str]] = None) -> None:
    """
    Renderiza um card para agrupar conteúdo (Gestalt: região comum + proximidade).

    Args:
        title: Título do card.
        body_md: Conteúdo em Markdown/HTML simples.
        pills: Pequenos rótulos para reforçar a leitura (similaridade).
    """
    pills_html = ""
    if pills:
        pills_html = "".join([f"<span class='miguel-pill'>{p}</span>" for p in pills])

    st.markdown(
        f"""
        <div class="miguel-card">
          <h3>{title}</h3>
          {pills_html}
          <div style="margin-top:0.45rem">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Estado da sessão e persistência local
# =============================================================================

DATA_DIR = "data"
CUSTOM_DOCS_PATH = os.path.join(DATA_DIR, "custom_docs.pkl")


def init_session_state() -> None:
    """Inicializa variáveis em st.session_state para consistência e prevenção de erros."""
    st.session_state.setdefault("page", "Chat")
    st.session_state.setdefault("query_history", [])
    st.session_state.setdefault("docs", [])
    st.session_state.setdefault("qa_chain", None)
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("embeddings", None)

    # Parâmetros interativos
    st.session_state.setdefault("retriever_k", 3)
    st.session_state.setdefault("max_new_tokens", 512)
    st.session_state.setdefault("temperature", 0.7)

    st.session_state.setdefault("toast", None)


def save_custom_docs(docs_list: List[str]) -> bool:
    """Salva a lista de documentos customizados localmente em arquivo pickle."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CUSTOM_DOCS_PATH, "wb") as f:
            pickle.dump(docs_list, f)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar documentos: {e}")
        return False


def load_custom_docs() -> Optional[List[str]]:
    """Carrega documentos customizados salvos, se existirem."""
    try:
        if os.path.exists(CUSTOM_DOCS_PATH):
            with open(CUSTOM_DOCS_PATH, "rb") as f:
                return pickle.load(f)
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar documentos: {e}")
        return None


# =============================================================================
# Conteúdo didático (documentos padrão + glossário)
# =============================================================================

def get_default_docs() -> List[str]:
    """Retorna uma base mínima de documentos de exemplo para o pipeline."""
    return [
        "Churn é o cancelamento/abandono de clientes em um serviço. É uma métrica importante de retenção.",
        "NPS (Net Promoter Score) mede lealdade perguntando se o cliente recomendaria a empresa; varia de -100 a +100.",
        "LangChain é um framework para construir aplicações com modelos de linguagem e componentes (memória, ferramentas, dados).",
        "RAG (Retrieval-Augmented Generation) conecta um modelo de linguagem a uma base de conhecimento para responder com evidências.",
        "Representações vetoriais (embeddings) transformam texto em números, preservando o significado para buscas por similaridade.",
        "Um pipeline RAG típico: dividir texto (chunking), gerar embeddings, indexar, recuperar trechos relevantes e gerar resposta.",
        "Machine Learning é uma área da IA em que sistemas aprendem padrões a partir de dados para tomar decisões ou fazer previsões.",
        "Deep Learning usa redes neurais com muitas camadas para lidar com padrões complexos (texto, imagem, áudio).",
        "NLP (Processamento de Linguagem Natural) é a área da IA que lida com compreensão e geração de linguagem humana.",
        "BI (Business Intelligence) reúne práticas e ferramentas para análise de dados e suporte à decisão em negócios.",
    ]


GLOSSARY: Dict[str, Dict[str, str]] = {
    "LLM (Large Language Model)": {
        "o_que_e": "Um modelo de linguagem em grande escala: aprende padrões de texto e consegue gerar respostas em linguagem natural.",
        "onde_aparece_no_app": "É o componente que gera a resposta final (aqui: FLAN-T5).",
        "por_que_importa": "Define a qualidade do texto gerado, mas pode ‘alucinar’ — por isso usamos RAG com documentos.",
    },
    "RAG (Geração aumentada por recuperação)": {
        "o_que_e": "Arquitetura que busca trechos relevantes em documentos e inclui esse contexto antes do modelo gerar a resposta.",
        "onde_aparece_no_app": "No ‘retriever’ + ‘vector store’, que selecionam os trechos e os entregam ao modelo.",
        "por_que_importa": "Aumenta a chance de resposta correta e ancorada em evidências (reduz alucinações).",
    },
    "Chunking (Divisão em trechos)": {
        "o_que_e": "Processo de dividir textos longos em partes menores para indexação e busca.",
        "onde_aparece_no_app": "Ao processar PDFs: o texto é cortado em trechos com sobreposição.",
        "por_que_importa": "Trechos menores tornam a busca por similaridade mais eficiente e o contexto mais útil.",
    },
    "Representações vetoriais (Embeddings)": {
        "o_que_e": "Transformação do texto em um vetor numérico que “representa o significado”.",
        "onde_aparece_no_app": "Usadas para indexar documentos e comparar similaridade com a pergunta.",
        "por_que_importa": "Permite busca semântica (por sentido), não apenas por palavras exatas.",
    },
    "FAISS (Vector Store)": {
        "o_que_e": "Biblioteca de busca eficiente por similaridade entre vetores.",
        "onde_aparece_no_app": "Armazena embeddings e retorna os trechos mais próximos da pergunta.",
        "por_que_importa": "Acelera a recuperação de informações mesmo com milhares de trechos.",
    },
    "Retriever (Recuperador)": {
        "o_que_e": "Componente que consulta o índice vetorial e retorna os Top-k trechos mais relevantes.",
        "onde_aparece_no_app": "Configuração ‘k’ (Top-k) influencia quantos trechos são enviados ao modelo.",
        "por_que_importa": "Poucos trechos → pode faltar contexto; muitos trechos → pode confundir o modelo.",
    },
    "Top-k (k)": {
        "o_que_e": "Quantidade de trechos retornados pela busca por similaridade.",
        "onde_aparece_no_app": "Configuração na barra lateral (Configurações do pipeline).",
        "por_que_importa": "Equilibra contexto suficiente e ruído excessivo.",
    },
    "Temperatura (temperature)": {
        "o_que_e": "Controla aleatoriedade do texto gerado: menor = mais previsível; maior = mais criativo.",
        "onde_aparece_no_app": "Configuração na barra lateral (Configurações do pipeline).",
        "por_que_importa": "Em contexto didático, valores menores tendem a gerar respostas mais consistentes.",
    },
    "max_new_tokens": {
        "o_que_e": "Limite máximo do tamanho da resposta (em tokens).",
        "onde_aparece_no_app": "Configuração na barra lateral (Configurações do pipeline).",
        "por_que_importa": "Controla tempo/uso de recursos e evita respostas longas demais.",
    },
}


# =============================================================================
# Métricas e segurança (recursos locais)
# =============================================================================

def get_system_stats() -> Dict[str, float]:
    """Coleta estatísticas básicas do sistema e do índice vetorial."""
    ram = psutil.virtual_memory()

    faiss_size_mb = 0.0
    total_vectors = 0
    total_docs = len(st.session_state.docs) if st.session_state.get("docs") else 0

    if st.session_state.get("vectorstore") is not None:
        try:
            faiss_index = st.session_state.vectorstore.index
            total_vectors = int(faiss_index.ntotal)
            faiss_size_mb = (total_vectors * 4) / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Erro ao obter dados do FAISS: {e}")

    return {
        "ram_used_gb": ram.used / (1024 ** 3),
        "ram_total_gb": ram.total / (1024 ** 3),
        "faiss_size_mb": float(faiss_size_mb),
        "total_vectors": float(total_vectors),
        "total_docs": float(total_docs),
    }


def check_system_safety() -> Dict[str, Any]:
    """Verifica limites “seguros” de uso para evitar travamentos por excesso de RAM/índice."""
    stats = get_system_stats()
    ram_usage_pct = stats["ram_used_gb"] / stats["ram_total_gb"] if stats["ram_total_gb"] else 0.0
    faiss_size_gb = stats["faiss_size_mb"] / 1024.0

    RAM_USAGE_THRESHOLD = 0.85
    FAISS_SIZE_THRESHOLD_GB = 8.0

    return {
        "ram_usage_pct": ram_usage_pct,
        "faiss_size_gb": faiss_size_gb,
        "ram_safe": ram_usage_pct < RAM_USAGE_THRESHOLD,
        "faiss_safe": faiss_size_gb < FAISS_SIZE_THRESHOLD_GB,
        "overall_safe": (ram_usage_pct < RAM_USAGE_THRESHOLD) and (faiss_size_gb < FAISS_SIZE_THRESHOLD_GB),
    }


def get_theoretical_limits() -> Dict[str, Any]:
    """Limites aproximados (didáticos) para orientar o usuário."""
    return {
        "max_vectors_estimate": 1_000_000,
        "max_docs_estimate": 20_000,
        "max_faiss_size_gb": 10.0,
        "context_window_tokens_estimate": 2048,
    }


# =============================================================================
# PDF -> texto -> chunks
# =============================================================================

def process_pdf(uploaded_file) -> List[Any]:
    """Extrai texto de um PDF e divide em trechos (chunks) com sobreposição."""
    tmp_file_path = None
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        return splitter.split_documents(pages)

    except Exception as e:
        logger.error(f"Erro ao processar PDF: {e}")
        raise
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Erro ao deletar temporário: {e}")


def process_pdf_safely(uploaded_file, max_chunks_per_file: int = 200) -> Tuple[List[Any], List[str]]:
    """Processa PDF com segurança limitando a quantidade de chunks."""
    warnings: List[str] = []
    try:
        loaded_docs = process_pdf(uploaded_file)
        if len(loaded_docs) > max_chunks_per_file:
            warnings.append(f"Arquivo grande: cortado para {max_chunks_per_file} trechos.")
            return loaded_docs[:max_chunks_per_file], warnings
        return loaded_docs, warnings
    except Exception as e:
        warnings.append(f"Erro ao processar PDF: {str(e)}")
        return [], warnings


# =============================================================================
# Pipeline RAG (LangChain + HuggingFace + FAISS)
# =============================================================================

def format_response(response_data: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Normaliza a resposta do pipeline RAG para exibição (texto + documentos fonte)."""
    if isinstance(response_data, dict):
        answer = response_data.get("result", "Resposta não encontrada.")
        source_docs = response_data.get("source_documents", [])
    else:
        answer = str(response_data)
        source_docs = []

    answer = answer.strip()
    if not answer:
        answer = "Desculpe, não consegui gerar uma resposta adequada para sua pergunta."
    return answer, source_docs


@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline(
    docs_texts: List[str],
    retriever_k: int,
    temperature: float,
    max_new_tokens: int,
):
    """
    Inicializa o pipeline RAG (com cache).

    Returns:
        (qa_chain, vectorstore, retriever, embeddings)
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
        from langchain_community.llms import HuggingFacePipeline

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = FAISS.from_texts(docs_texts, embeddings)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(retriever_k)},
        )

        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        gen_pipeline = hf_pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True,
            repetition_penalty=1.1,
        )

        llm = HuggingFacePipeline(pipeline=gen_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
        )

        return qa_chain, vectorstore, retriever, embeddings

    except Exception as e:
        logger.error(f"Erro ao inicializar pipeline: {e}")
        return None, None, None, None


def ensure_pipeline_ready() -> None:
    """Garante que o pipeline esteja inicializado no estado da sessão."""
    if st.session_state.qa_chain is not None:
        return

    custom_docs = load_custom_docs()
    docs = custom_docs if custom_docs else get_default_docs()

    with st.spinner("Inicializando o modelo e a base de conhecimento…"):
        qa_chain, vectorstore, retriever, embeddings = initialize_rag_pipeline(
            docs_texts=docs,
            retriever_k=st.session_state.retriever_k,
            temperature=st.session_state.temperature,
            max_new_tokens=st.session_state.max_new_tokens,
        )

    if qa_chain is None:
        st.error("Não foi possível inicializar o pipeline. Verifique dependências e tente novamente.")
        st.stop()

    st.session_state.qa_chain = qa_chain
    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.embeddings = embeddings
    st.session_state.docs = docs


# =============================================================================
# Sidebar (navegação, configurações, status)
# =============================================================================

def render_sidebar() -> None:
    """Barra lateral com navegação, configurações e status do sistema."""
    with st.sidebar:
        st.markdown(f"### {APP_SHORT}")
        st.caption("Navegação e configurações")

        st.session_state.page = st.radio(
            "Ir para",
            options=["Chat", "Documentos", "Glossário & Ajuda"],
            index=["Chat", "Documentos", "Glossário & Ajuda"].index(st.session_state.page),
            label_visibility="collapsed",
        )

        st.markdown("---")

        with st.expander("Configurações do pipeline", expanded=True):
            st.caption("Ajustes que afetam busca e geração. Veja o significado no Glossário.")

            new_k = st.slider("Top-k (k): trechos recuperados", 1, 8, int(st.session_state.retriever_k))
            new_temp = st.slider("Temperatura (criatividade)", 0.0, 1.2, float(st.session_state.temperature), 0.05)
            new_tokens = st.slider("Tamanho máximo da resposta (tokens)", 64, 1024, int(st.session_state.max_new_tokens), 32)

            cols = st.columns(2)
            apply_clicked = cols[0].button("Aplicar", use_container_width=True)
            reset_clicked = cols[1].button("Padrão", use_container_width=True)

            if reset_clicked:
                st.session_state.retriever_k = 3
                st.session_state.temperature = 0.7
                st.session_state.max_new_tokens = 512
                st.cache_resource.clear()
                st.session_state.qa_chain = None
                st.session_state.toast = "Configurações restauradas para o padrão."
                st.rerun()

            if apply_clicked:
                st.session_state.retriever_k = int(new_k)
                st.session_state.temperature = float(new_temp)
                st.session_state.max_new_tokens = int(new_tokens)

                st.cache_resource.clear()
                st.session_state.qa_chain = None
                st.session_state.toast = "Configurações aplicadas. O pipeline será recarregado."
                st.rerun()

        st.markdown("---")

        stats = get_system_stats()
        safety = check_system_safety()

        status_color = "state-ok" if safety["overall_safe"] else "state-warn"
        st.markdown(f"**Status do sistema:** <span class='{status_color}'>●</span>", unsafe_allow_html=True)
        st.caption("Recursos locais (para evitar travamentos).")

        st.write(f"RAM: {stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.1f} GB")
        st.write(f"Documentos indexados: {len(st.session_state.docs):,}")
        st.write(f"Vetores (FAISS): {int(stats['total_vectors']):,}")

        if not safety["overall_safe"]:
            st.warning("Uso alto de recursos. Considere reduzir PDFs ou quantidade de trechos.")


# =============================================================================
# Páginas
# =============================================================================

def page_chat() -> None:
    """Página de chat (pergunta -> resposta + evidências)."""
    render_header()

    if st.session_state.toast:
        st.info(st.session_state.toast)
        st.session_state.toast = None

    col_left, col_right = st.columns([2.2, 1])

    with col_left:
        card(
            "Pergunte ao modelo",
            """
            Use perguntas curtas e diretas. Para respostas mais “ancoradas”, suba PDFs na aba **Documentos**.
            """,
            pills=["RAG", "LLM local", "Evidências"],
        )

        query = st.text_area(
            "Pergunta",
            placeholder="Ex.: O que é RAG? Como embeddings ajudam na busca? O que é Deep Learning?",
            height=110,
            key="chat_input",
            label_visibility="collapsed",
        )

        st.markdown("**Sugestões (clique para preencher):**")
        btn_cols = st.columns(3)

        examples = [
            ("O que é RAG?", "Explique o que é RAG e por que ele ajuda a reduzir alucinações."),
            ("Embeddings", "O que são representações vetoriais (embeddings) e para que servem?"),
            ("FAISS", "O que é FAISS e como ele ajuda a encontrar documentos similares?"),
        ]

        for i, (label, text) in enumerate(examples):
            if btn_cols[i].button(label, use_container_width=True):
                st.session_state["chat_input"] = text
                st.rerun()

        ask = st.button("Buscar resposta", type="primary", use_container_width=True)

        if ask:
            if not query.strip():
                st.warning("Digite uma pergunta antes de buscar.")
                return

            safety = check_system_safety()
            if not safety["overall_safe"]:
                st.error("Recursos do sistema estão altos. Reduza documentos/trechos e tente novamente.")
                return

            with st.spinner("Processando… (buscando trechos relevantes e gerando resposta)"):
                try:
                    start = time.time()
                    response = st.session_state.qa_chain.invoke({"query": query})
                    elapsed = time.time() - start

                    answer, source_docs = format_response(response)

                    st.session_state.query_history.append(
                        {
                            "query": query,
                            "answer": answer,
                            "time": elapsed,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )

                    card(
                        "Resposta",
                        f"""
                        <b>Texto gerado:</b><br/>
                        {answer}
                        <br/><br/>
                        <span class="miguel-pill">Tempo: {elapsed:.2f}s</span>
                        """,
                        pills=["Resposta", "Tempo", "Clareza"],
                    )

                    if source_docs:
                        st.subheader("Evidências utilizadas (trechos recuperados)")
                        st.caption("Mostramos os trechos que mais contribuíram para a resposta.")

                        for idx, doc in enumerate(source_docs[: min(3, len(source_docs))], 1):
                            with st.expander(f"Trecho {idx}", expanded=(idx == 1)):
                                st.write(doc.page_content)

                except Exception as e:
                    logger.error(f"Erro ao responder: {e}")
                    st.error("Ocorreu um erro ao gerar a resposta. Tente novamente ou reduza documentos.")

    with col_right:
        history = st.session_state.query_history
        total = len(history)

        # Evita formatação de None como float
        if total:
            avg = sum(x["time"] for x in history) / total
            last = history[-1]["time"]
            avg_txt = f"{avg:.2f}s"
            last_txt = f"{last:.2f}s"
        else:
            avg_txt = "-"
            last_txt = "-"

        card(
            "Métricas da sessão",
            f"""
            - Total de perguntas: <b>{total}</b><br/>
            - Tempo médio: <b>{avg_txt}</b><br/>
            - Última pergunta: <b>{last_txt}</b>
            """,
            pills=["Observação", "Transparência"],
        )

        card(
            "Base de conhecimento",
            f"""
            - Documentos indexados: <b>{len(st.session_state.docs)}</b><br/>
            - Top-k atual: <b>{st.session_state.retriever_k}</b><br/><br/>
            Dica: envie PDFs na aba <b>Documentos</b> para enriquecer a base.
            """,
            pills=["Docs", "Top-k"],
        )

    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("Histórico recente")
        for item in reversed(st.session_state.query_history[-5:]):
            with st.expander(f"{item['timestamp']} — {item['query'][:60]}"):
                st.write(f"**Pergunta:** {item['query']}")
                st.write(f"**Resposta:** {item['answer']}")
                st.write(f"**Tempo:** {item['time']:.2f}s")


def page_documents() -> None:
    """Página de upload/processamento de PDFs e atualização do pipeline."""
    render_header()

    card(
        "Gerenciar base de conhecimento",
        """
        Envie PDFs para ampliar o conhecimento do sistema. O texto será dividido em trechos (chunking),
        transformado em representações vetoriais (embeddings) e indexado no FAISS para busca por similaridade.
        """,
        pills=["PDF", "Chunking", "FAISS"],
    )

    uploaded_files = st.file_uploader(
        "Enviar PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Dica: comece com 1 PDF pequeno para observar o comportamento do RAG.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} arquivo(s) selecionado(s).")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

        process = st.button("Processar e adicionar à base", type="primary", use_container_width=True)

        if process:
            safety = check_system_safety()
            if not safety["overall_safe"]:
                st.error("Recursos do sistema estão altos. Envie PDFs menores ou feche outros apps.")
                return

            progress = st.progress(0)
            status = st.empty()

            new_texts: List[str] = []
            warnings_all: List[str] = []

            for i, f in enumerate(uploaded_files, start=1):
                status.text(f"Processando: {f.name}")
                progress.progress(int((i - 0.25) / len(uploaded_files) * 100) / 100)

                docs, warns = process_pdf_safely(f, max_chunks_per_file=200)
                warnings_all.extend([f"{f.name}: {w}" for w in warns])

                for d in docs:
                    new_texts.append(d.page_content)

                safety_now = check_system_safety()
                if not safety_now["ram_safe"]:
                    warnings_all.append("Uso de RAM alto: interrompemos o processamento para evitar travamento.")
                    break

                progress.progress(int(i / len(uploaded_files) * 100) / 100)

            if warnings_all:
                with st.expander("Avisos do processamento"):
                    for w in warnings_all:
                        st.warning(w)

            if not new_texts:
                st.warning("Nenhum texto foi extraído. Tente outro PDF.")
                return

            status.text("Atualizando base e reinicializando pipeline…")
            updated_docs = (st.session_state.docs or []) + new_texts

            if save_custom_docs(updated_docs):
                st.cache_resource.clear()
                st.session_state.qa_chain = None
                st.session_state.docs = updated_docs
                ensure_pipeline_ready()
                status.text("Concluído.")
                st.success(f"{len(new_texts)} trechos adicionados à base.")
            else:
                st.error("Não foi possível salvar os documentos. Verifique permissões de escrita.")
                return

    st.markdown("---")

    stats = get_system_stats()
    limits = get_theoretical_limits()

    cols = st.columns(3)
    cols[0].metric("Documentos (strings)", int(stats["total_docs"]))
    cols[1].metric("Vetores (FAISS)", int(stats["total_vectors"]))
    cols[2].metric("FAISS (estimado)", f"{stats['faiss_size_mb']:.1f} MB")

    with st.expander("Ver amostra dos documentos indexados"):
        docs = st.session_state.docs or []
        for i, d in enumerate(docs[:8], 1):
            st.write(f"{i}. {d[:180]}{'…' if len(d) > 180 else ''}")
        if len(docs) > 8:
            st.caption(f"… e mais {len(docs) - 8} documento(s).")

    with st.expander("Limites (referências didáticas)"):
        st.write(f"- Máx. vetores (estimativa): {limits['max_vectors_estimate']:,}")
        st.write(f"- Máx. documentos (estimativa): {limits['max_docs_estimate']:,}")
        st.write(f"- Máx. FAISS (estimativa): {limits['max_faiss_size_gb']:.1f} GB")
        st.write(f"- Janela de contexto (estimativa): {limits['context_window_tokens_estimate']} tokens")

    st.markdown("---")

    st.subheader("Gerenciamento avançado")
    col_a, col_b = st.columns(2)

    if col_a.button("Recarregar pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.qa_chain = None
        st.session_state.toast = "Pipeline recarregado."
        st.rerun()

    with col_b:
        confirm = st.checkbox("Confirmo que desejo voltar para a base padrão (isso remove meus PDFs)")
        if st.button("Resetar base", use_container_width=True, disabled=not confirm):
            try:
                if os.path.exists(CUSTOM_DOCS_PATH):
                    os.remove(CUSTOM_DOCS_PATH)
                st.cache_resource.clear()
                st.session_state.qa_chain = None
                st.session_state.docs = get_default_docs()
                st.session_state.toast = "Base resetada para o padrão."
                st.rerun()
            except Exception as e:
                logger.error(f"Erro ao resetar base: {e}")
                st.error("Não foi possível resetar. Verifique permissões de arquivo.")


def page_glossary_help() -> None:
    """Página de ajuda: glossário + heurísticas de Nielsen + Gestalt."""
    render_header()

    card(
        "Glossário (termos técnicos do aplicativo)",
        """
        Selecione um termo e veja <b>o que é</b>, <b>onde aparece no app</b> e <b>por que importa</b>.
        A ideia é permitir exploração sem depender de jargões.
        """,
        pills=["Didático", "Autoexplicativo"],
    )

    term = st.selectbox("Escolha um termo", list(GLOSSARY.keys()))
    info = GLOSSARY[term]

    card(
        term,
        f"""
        <b>O que é:</b> {info["o_que_e"]}<br/><br/>
        <b>Onde aparece no aplicativo:</b> {info["onde_aparece_no_app"]}<br/><br/>
        <b>Por que é importante:</b> {info["por_que_importa"]}
        """,
        pills=["Definição", "Uso no app", "Importância"],
    )

    st.markdown("---")

    card(
        "Heurísticas de Nielsen (como melhoramos a usabilidade)",
        """
        <b>Visibilidade do status:</b> spinners, progresso e métricas (tempo, docs, vetores).<br/>
        <b>Correspondência com o mundo real:</b> termos simples (“trechos”, “evidências”, “pergunta”).<br/>
        <b>Controle e liberdade:</b> reset com confirmação e recarregar pipeline.<br/>
        <b>Consistência e padrões:</b> navegação fixa e rótulos uniformes.<br/>
        <b>Prevenção de erros:</b> limites de chunks e checagem de recursos.<br/>
        <b>Reconhecimento em vez de memorização:</b> exemplos clicáveis e histórico.<br/>
        <b>Design minimalista:</b> foco no essencial e bom contraste.<br/>
        <b>Ajuda e documentação:</b> glossário e textos de apoio.
        """,
        pills=["Nielsen", "UX", "Didática"],
    )

    card(
        "Princípios da Gestalt (o que você vê na interface)",
        """
        <b>Proximidade:</b> itens relacionados ficam juntos (ex.: configurações do pipeline).<br/>
        <b>Similaridade:</b> cards e blocos com o mesmo estilo, facilitando leitura rápida.<br/>
        <b>Região comum:</b> bordas e fundos agrupam conceitos (ex.: “Resposta” e “Evidências”).<br/>
        <b>Figura-fundo:</b> contraste alto melhora legibilidade.<br/>
        <b>Continuidade:</b> fluxo de leitura vertical (Pergunta → Resposta → Evidências → Histórico).<br/>
        <b>Fechamento:</b> expanders exibem detalhes sem poluir a tela.
        """,
        pills=["Gestalt", "Layout", "Percepção"],
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Ponto de entrada do aplicativo."""
    inject_minimal_css()
    init_session_state()
    render_sidebar()

    ensure_pipeline_ready()

    if st.session_state.page == "Chat":
        page_chat()
    elif st.session_state.page == "Documentos":
        page_documents()
    else:
        page_glossary_help()

    st.markdown("---")
    st.caption(
        f"{APP_SHORT} — aplicação didática, local e gratuita | "
        "Tecnologias: LangChain, HuggingFace, FAISS, Streamlit"
    )


if __name__ == "__main__":
    main()
