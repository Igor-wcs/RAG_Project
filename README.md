# 🤖 Agente de RH Inteligente: RAG Avançado com Semantic Reranking

Este repositório contém uma aplicação de **Retrieval-Augmented Generation (RAG)** desenvolvida para atuar como um assistente de políticas de RH. O projeto demonstra a implementação de um pipeline de recuperação de documentos em duas etapas, utilizando LLMs de última geração para garantir a precisão das respostas corporativas.

## 🚀 Diferenciais Técnicos

Diferente de implementações básicas de RAG, este projeto foca na qualidade do contexto recuperado através de:

* **Enriquecimento de Metadados:** Os chunks de texto são classificados semanticamente (ex: "férias", "conduta", "home office") durante a ingestão, permitindo uma rastreabilidade superior das fontes.
* **Recuperação em Duas Etapas (Retrieve & Rerank):** 1.  **Busca por Similaridade:** Recupera os 8 trechos mais próximos usando busca vetorial.
    2.  **Reranking Semântico:** Utiliza o modelo `gpt-4o-mini` como "juiz" para atribuir notas de 0 a 10 à relevância de cada trecho, filtrando apenas os 4 melhores para a geração final.
* **Armazenamento Vetorial Persistente:** Implementação com **ChromaDB**, garantindo que o índice vetorial seja salvo localmente no diretório `./chroma_rh` para eficiência de custo e tempo.

## 🛠️ Stack Tecnológica

* **Modelos:** OpenAI `gpt-4o-mini` (LLM) e `text-embedding-3-small` (Embeddings).
* **Orquestração:** [LangChain](https://www.langchain.com/).
* **Interface:** Streamlit para uma experiência de usuário fluida.
* **Banco de Dados:** ChromaDB para armazenamento de vetores.
* **Processamento de PDF:** PyPDFLoader e RecursiveCharacterTextSplitter.

## 📐 Arquitetura do Pipeline

1.  **Ingestão:** Os documentos (como o Código de Conduta da Nestlé) são carregados e divididos em chunks de 800 caracteres com sobreposição de 150 caracteres para manter a coesão.
2.  **Indexação:** Cada chunk recebe metadados categoriais antes de ser convertido em vetores.
3.  **Processamento de Query:** A pergunta do usuário é enviada ao banco vetorial.
4.  **Refino:** O algoritmo de reranking limpa o contexto de ruídos irrelevantes.
5.  **Geração:** O modelo gera uma resposta baseada estritamente no contexto fornecido, assumindo a persona de um agente de RH.

## ⚙️ Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/rh-rag-agent.git](https://github.com/seu-usuario/rh-rag-agent.git)
    cd rh-rag-agent
    ```

2.  **Instale as dependências:**
    ```bash
    pip install streamlit langchain langchain-openai chromadb pypdf
    ```

3.  **Configuração da API:**
    Certifique-se de configurar sua chave da OpenAI como variável de ambiente ou no arquivo de configuração (evite hardcoding em produção).

4.  **Rode a aplicação:**
    ```bash
    streamlit run Rag_Project.py
    ```

---


---

*Este projeto foi desenvolvido para fins de demonstração técnica de fluxos avançados de LLMs e RAG.*