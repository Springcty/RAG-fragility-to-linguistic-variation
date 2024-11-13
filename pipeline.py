import torch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Embedding model setup
embedding_model_name = 'nvidia/NV-Embed-v2'

model_kwargs = {
    'device': 'cuda',
    'trust_remote_code': True,
    'model_kwargs': {'torch_dtype': torch.float16},
    'prompts': {'passage': 'Given a web search query, retrieve relevant passages that answer the query.\n', 
                'query': 'query: '}
    }
encode_kwargs = {'normalize_embeddings': True}

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# TODO: Load document corpus as a list of documents: docs

# Chroma database setup
chroma_db_path = '/data/user_data/tianyuca/chroma_db/' # TODO
vectorstore = Chroma.from_texts(
    texts=docs,
    embedding=embedding_model,
    persist_directory=chroma_db_path,
)

# Retriever setup
retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) # retrieve top 5 most relevant documents


# Load tokenizer and model
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
model.eval()

# Set up generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)
llm = HuggingFacePipeline(pipeline=generator)

# Set up the RAG chain
system_prompt = (
    'You are a professional question-answer task assistant. Use the following pieces of retrieved context to answer the question briefly.'
    'Context: {context}'
    )
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', 'Question: {input}'),
    ]
)
questions_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, questions_answer_chain)

# Answer generation
rag_chain.invoke({'input': query})