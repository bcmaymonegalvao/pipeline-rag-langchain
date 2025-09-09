import streamlit as st
import time
import os
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="🚀 Pipeline RAG LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    """Inicializa o pipeline RAG com cache para otimizar performance"""
    try:
        with st.spinner("🔧 Inicializando pipeline RAG..."):
            # Importações necessárias - ordem correta
            import torch
            from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            
            # Importar transformers por último para evitar conflitos
            from transformers import (
                AutoTokenizer, 
                AutoModelForSeq2SeqLM, 
                pipeline as hf_pipeline  # Renomear para evitar conflito
            )
            from langchain_community.llms import HuggingFacePipeline
            
            # Documentos de exemplo (podem ser expandidos)
            docs = [
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
            
            # Configurar embeddings
            st.info("📊 Carregando modelo de embeddings MiniLM...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Criar índice FAISS
            st.info("🗃️ Construindo índice vetorial FAISS...")
            vectorstore = FAISS.from_texts(docs, embeddings)
            
            # Configurar retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Aumentamos para 3 para mais contexto
            )
            
            # Configurar LLM local
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
            
            # Criar chain RAG
            st.info("🔗 Configurando chain RAG...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,  # Para mostrar fontes
                verbose=False
            )
            
            st.success("✅ Pipeline RAG inicializado com sucesso!")
            
            return qa_chain, vectorstore, docs, retriever
            
    except Exception as e:
        st.error(f"❌ Erro ao inicializar pipeline: {str(e)}")
        logger.error(f"Erro na inicialização: {e}")
        return None, None, None, None

def format_response(response_data: Dict[str, Any]) -> tuple:
    """Formata a resposta do RAG para exibição"""
    if isinstance(response_data, dict):
        answer = response_data.get("result", "Resposta não encontrada")
        source_docs = response_data.get("source_documents", [])
    else:
        answer = str(response_data)
        source_docs = []
    
    # Limpar resposta
    answer = answer.strip()
    if not answer or answer.lower() in ["", "none", "null"]:
        answer = "Desculpe, não consegui gerar uma resposta adequada para sua pergunta."
    
    return answer, source_docs

def main():
    # Header principal
    st.markdown('<div class="main-header">🚀 Pipeline RAG LangChain</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Retrieval-Augmented Generation com HuggingFace & FAISS</div>', unsafe_allow_html=True)
    
    # Sidebar com informações
    with st.sidebar:
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
        - 📚 Base de conhecimento expandível
        """)
        
        st.header("📋 Como usar")
        st.markdown("""
        1. **Digite sua pergunta** no campo abaixo
        2. **Clique em "Buscar Resposta"**
        3. **Veja a resposta** e documentos fonte
        4. **Explore diferentes temas**: IA, ML, NLP, BI
        """)
        
        # Estatísticas do sistema
        st.header("📊 Sistema")
        if 'qa_chain' in st.session_state:
            st.success("🟢 Pipeline Ativo")
            st.info(f"📄 Documentos: {len(st.session_state.docs)}")
            st.info("🔍 Retriever: Top-3")
        else:
            st.warning("🟡 Inicializando...")

    # Inicializar pipeline (apenas uma vez)
    if 'qa_chain' not in st.session_state:
        qa_chain, vectorstore, docs, retriever = initialize_rag_pipeline()
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.vectorstore = vectorstore
            st.session_state.docs = docs
            st.session_state.retriever = retriever
            st.session_state.query_history = []
        else:
            st.stop()
    
    # Interface principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💭 Faça sua pergunta")
        
        # Campo de entrada da pergunta
        query = st.text_area(
            "Digite sua pergunta aqui:",
            placeholder="Ex: O que é RAG? Como funciona machine learning? O que significa churn?",
            height=100,
            help="Digite qualquer pergunta sobre IA, ML, NLP, BI ou conceitos relacionados"
        )
        
        # Botões de exemplo
        st.markdown("**💡 Perguntas de exemplo:**")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 O que é churn?"):
                query = "O que significa churn em análise de clientes?"
        
        with col_btn2:
            if st.button("🤖 Como funciona RAG?"):
                query = "Como funciona o pipeline de RAG?"
        
        with col_btn3:
            if st.button("📚 O que é LangChain?"):
                query = "Explique o que é LangChain e para que serve."
        
        # Botão de busca
        search_button = st.button("🔍 Buscar Resposta", type="primary", use_container_width=True)
        
        # Processar pergunta
        if search_button and query.strip():
            with st.spinner("🤔 Processando sua pergunta..."):
                try:
                    # Registrar tempo de resposta
                    start_time = time.time()
                    
                    # Executar RAG
                    response = st.session_state.qa_chain.invoke({"query": query})
                    
                    # Calcular tempo
                    response_time = time.time() - start_time
                    
                    # Formatar resposta
                    answer, source_docs = format_response(response)
                    
                    # Adicionar ao histórico
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": answer,
                        "time": response_time,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    
                    # Exibir resposta
                    st.markdown('<div class="response-card">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 Resposta:**\n\n{answer}")
                    st.markdown(f"⏱️ *Tempo de resposta: {response_time:.2f}s*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Mostrar documentos fonte
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
        
        # Métricas
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
        
        # Base de conhecimento
        st.header("📚 Base de Conhecimento")
        st.info(f"📄 {len(st.session_state.docs)} documentos indexados")
        
        with st.expander("👁️ Ver documentos"):
            for i, doc in enumerate(st.session_state.docs, 1):
                st.markdown(f"**{i}.** {doc[:100]}...")
    
    # Histórico de perguntas
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.header("📋 Histórico de Perguntas")
        
        # Mostrar apenas as últimas 5 perguntas
        recent_history = st.session_state.query_history[-5:]
        
        for item in reversed(recent_history):
            with st.expander(f"🕐 {item['timestamp']} - {item['query'][:50]}..."):
                st.markdown(f"**Pergunta:** {item['query']}")
                st.markdown(f"**Resposta:** {item['answer']}")
                st.markdown(f"**Tempo:** {item['time']:.2f}s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #636e72;'>
        🚀 Desenvolvido com Streamlit | 🤖 Powered by HuggingFace & LangChain
        <br>
        💡 Sistema RAG 100% gratuito e local
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()