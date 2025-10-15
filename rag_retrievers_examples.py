"""
ADVANCED RAG RETRIEVERS - EJEMPLOS PRÁCTICOS
Curso: IBM RAG and Agentic AI Professional Certificate
"""

# =============================================================================
# 1. GENERAL Q&A: Vector + BM25 (Hybrid Search)
# =============================================================================

# --- Usando LangChain ---
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Documentos de ejemplo
docs = [
    Document(page_content="Los motores de combustión interna funcionan quemando gasolina"),
    Document(page_content="El motor eléctrico usa baterías para generar movimiento"),
    Document(page_content="Los mecanismos de propulsión incluyen turbinas y pistones")
]

# Retriever 1: Vector (búsqueda semántica)
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Retriever 2: BM25 (búsqueda por palabras clave)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

# Combinar ambos (Hybrid)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 50% cada uno
)

# Usar
results = ensemble_retriever.get_relevant_documents("¿Cómo funcionan los motores?")


# =============================================================================
# 2. TECHNICAL DOCS: BM25 + Vector
# =============================================================================

# --- Usando LlamaIndex ---
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers import BM25Retriever
from llama_index.retrievers import QueryFusionRetriever

# Cargar documentación técnica
documents = SimpleDirectoryReader("./technical_docs").load_data()

# Crear índice
index = VectorStoreIndex.from_documents(documents)

# Retriever principal: BM25 (términos exactos)
bm25_retriever = BM25Retriever.from_defaults(
    index=index, 
    similarity_top_k=5
)

# Retriever secundario: Vector
vector_retriever = index.as_retriever(similarity_top_k=5)

# Fusionar (BM25 tiene prioridad)
retriever = QueryFusionRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    similarity_top_k=3,
    mode="reciprocal_rerank"  # Re-rankea resultados
)

# Buscar función específica
results = retriever.retrieve("función getUserData()")


# =============================================================================
# 3. LONG DOCUMENTS: Auto Merging Retriever
# =============================================================================

# --- Usando LlamaIndex ---
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.retrievers import AutoMergingRetriever
from llama_index import ServiceContext, VectorStoreIndex

# Documentos largos
documents = SimpleDirectoryReader("./long_docs").load_data()

# Parser jerárquico: crea chunks de diferentes tamaños
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Grande, medio, pequeño
)

# Parsear documentos
nodes = node_parser.get_nodes_from_documents(documents)

# Crear índice
service_context = ServiceContext.from_defaults()
index = VectorStoreIndex(nodes, service_context=service_context)

# Auto-merging retriever
retriever = AutoMergingRetriever(
    index.as_retriever(similarity_top_k=6),
    storage_context=index.storage_context,
    verbose=True
)

# Cuando encuentra chunks consecutivos relevantes, los fusiona automáticamente
results = retriever.retrieve("Explica el proceso completo de fotosíntesis")


# =============================================================================
# 4. RESEARCH PAPERS: Recursive Retriever
# =============================================================================

# --- Usando LlamaIndex ---
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.schema import IndexNode

# Crear estructura jerárquica de papers
# Nivel 1: Abstracts
abstract_index = VectorStoreIndex.from_documents(abstract_docs)

# Nivel 2: Secciones completas
section_indices = {}
for paper_id, sections in papers.items():
    section_indices[paper_id] = VectorStoreIndex.from_documents(sections)

# Crear nodos índice que apuntan a secciones
index_nodes = []
for paper_id, section_index in section_indices.items():
    node = IndexNode(
        text=f"Secciones del paper {paper_id}",
        index_id=paper_id
    )
    index_nodes.append(node)

# Recursive retriever
retriever = RecursiveRetriever(
    root_id="abstracts",
    retriever_dict={
        "abstracts": abstract_index.as_retriever(),
        **{pid: idx.as_retriever() for pid, idx in section_indices.items()}
    }
)

# Busca primero en abstracts, luego profundiza en secciones relevantes
results = retriever.retrieve("machine learning optimization techniques")


# =============================================================================
# 5. LARGE DOCUMENT SETS: Document Summary Index + Vector
# =============================================================================

# --- Usando LlamaIndex ---
from llama_index import DocumentSummaryIndex
from llama_index.llms import OpenAI

# Miles de documentos
documents = SimpleDirectoryReader("./large_corpus").load_data()

# LLM para generar resúmenes
llm = OpenAI(model="gpt-3.5-turbo")

# Crear índice con resúmenes automáticos
summary_index = DocumentSummaryIndex.from_documents(
    documents,
    service_context=ServiceContext.from_defaults(llm=llm),
    response_mode="tree_summarize"
)

# Retriever que primero busca en resúmenes
retriever = summary_index.as_retriever(
    retriever_mode="embedding",  # Busca en embeddings de resúmenes
    similarity_top_k=5
)

# Luego busca en documentos completos relevantes
query_engine = summary_index.as_query_engine(
    retriever=retriever,
    response_mode="compact"
)

# Eficiente incluso con miles de docs
response = query_engine.query("¿Qué reportes mencionan cambio climático?")


# =============================================================================
# COMPARACIÓN DE RENDIMIENTO
# =============================================================================

"""
Método                  | Velocidad | Precisión | Mejor para
------------------------|-----------|-----------|---------------------------
Vector + BM25           | ⭐⭐⭐     | ⭐⭐⭐⭐⭐   | Preguntas generales
BM25 + Vector           | ⭐⭐⭐⭐    | ⭐⭐⭐⭐    | Docs técnicos
Auto Merging            | ⭐⭐       | ⭐⭐⭐⭐⭐   | Documentos largos
Recursive               | ⭐⭐       | ⭐⭐⭐⭐⭐   | Papers académicos
Document Summary        | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐    | Miles de documentos
"""


# =============================================================================
# EJEMPLO COMPLETO: RAG con Hybrid Search
# =============================================================================

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Crear RAG completo con hybrid search
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,  # Del ejemplo 1
    return_source_documents=True
)

# Consultar
result = qa_chain({"query": "¿Cómo funcionan los motores eléctricos?"})

print("Respuesta:", result["result"])
print("\nFuentes:", result["source_documents"])


# =============================================================================
# TIPS IMPORTANTES
# =============================================================================

"""
1. HYBRID SEARCH (Vector + BM25):
   - Usa weights para ajustar balance
   - BM25 mejor para términos técnicos exactos
   - Vector mejor para conceptos y sinónimos

2. AUTO MERGING:
   - Ajusta chunk_sizes según longitud de docs
   - Más levels = más flexibilidad pero más memoria

3. RECURSIVE:
   - Perfecto para docs estructurados jerárquicamente
   - Define bien la jerarquía (abstract → section → subsection)

4. DOCUMENT SUMMARY:
   - Los resúmenes se generan UNA VEZ (cacheable)
   - Muy eficiente para consultas repetidas
   - Trade-off: calidad del resumen vs velocidad

5. EVALUACIÓN:
   - Siempre mide latencia y relevancia
   - Usa métricas: MRR, NDCG, Precision@K
   - Prueba con queries reales de tu dominio
"""
