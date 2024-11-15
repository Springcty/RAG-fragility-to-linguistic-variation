import argparse
import json
import os
# Set up LangchainSmith for trace
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = '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'
from langchain_core.tracers.context import tracing_v2_enabled

import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.retrievers import WikipediaRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from utils import load_linguistic_query


parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/user_data/tianyuca',
                    help='The path to load the dataset')
parser.add_argument('--result_path', type=str, default='/data/user_data/tianyuca/QL_result',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--dataset', type=str, default='PopQA',
                    help='Name of the QA dataset from ["PopQA", "EntityQuestions"]')
parser.add_argument('--property', type=str, default='Readability',
                    help='The linguistic properties of the query to be modified')
parser.add_argument('--embedding_model_name', type=str, default='nvidia/NV-Embed-v2',
                    help='Embedding model used for generating embeddings')
parser.add_argument('--customize_retriever', action='store_true',
                    help='True if use customized retriever instead of WikipediaRetriever from LangChain')
parser.add_argument('--model_name', type=str, default='/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct',
                    help='LLM used for generation')

args = parser.parse_args()

# Load the dataset
questions_o, questions_m, scores_o, scores_m, answers = load_linguistic_query(args)

# Set up the retriever
if args.customize_retriever:
    # Embedding model setup
    model_kwargs = {
        'device': 'cuda',
        'trust_remote_code': True,
        'model_kwargs': {'torch_dtype': torch.float16},
        }
    encode_kwargs = {'normalize_embeddings': True}

    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # TODO: Load document corpus as a list of documents: docs

    # Chroma database setup
    chroma_db_path = '/data/user_data/tianyuca/chroma_db/' # TODO
    vectorstore = Chroma.from_texts(
        texts=...,
        embedding=embedding_model,
        persist_directory=chroma_db_path,
    )

    # Retriever setup
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) # retrieve top 5 most relevant documents
else:
    retriever = WikipediaRetriever(top_k_results=3)
    
print('-'*10, 'Retriever setup successfully!', '-'*10)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to('cuda')
model.eval()
terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

# Set up generation pipeline from Huggingface
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    model_kwargs={'torch_dtype': torch.bfloat16},
    max_new_tokens=128,
    device='cuda',
)
llm = HuggingFacePipeline(
    pipeline=generator, 
    pipeline_kwargs={
        'eos_token_id': terminators,
        'pad_token_id': 128009,
        },
    )
print('-'*10, 'Generation model setup successfully!', '-'*10)

# Set up the RAG chain
system_prompt = (
    'You are a professional question-answer task assistant. Use the following pieces of retrieved context to answer the question briefly. Use one sentence and keep the answer concise.\n\n'
    'Context: {context}'
    )
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', 'Question: {input}\n\nAnswer:'),
    ]
)
questions_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, questions_answer_chain)
print('-'*10, 'RAG chain setup successfully!', '-'*10)
print('-'*10, 'Begin query!', '-'*10)

# Answer generation
rag_dict_o = {}
for i, q in enumerate(questions_o):
    with tracing_v2_enabled(project_name=f'{args.dataset}_{args.property}'):
        result = rag_chain.invoke({'input': q})

    retrieval_doc_id = [doc.metadata['source'] for doc in result['context']] # wiki link
    candidate_answer = result['answer']
    rag_dict_o[q] = {'retrieval_doc_id': retrieval_doc_id, 'candidate_answer': candidate_answer, 'answer': answers[i], 'score': scores_o[i]}

rag_dict_m = {}
for i, q in enumerate(questions_m):
    with tracing_v2_enabled(project_name=f'{args.dataset}_{args.property}'):
        result = rag_chain.invoke({'input': q})
    retrieval_doc_id = [doc.metadata['source'] for doc in result['context']] # wiki link
    candidate_answer = result['answer']
    rag_dict_m[q] = {'retrieval_doc_id': retrieval_doc_id, 'candidate_answer': candidate_answer, 'answer': answers[i], 'score': scores_m[i]}

save_path = f'{args.result_path}/{args.dataset}/{args.property}'
os.makedirs(save_path, exist_ok=True)
with open(f'{save_path}/results_original_query.json', 'w') as f:
    json.dump(rag_dict_o, f)
with open(f'{save_path}/results_modified_query.json', 'w') as f:
    json.dump(rag_dict_m, f)
print('-'*10, 'Save retrieval and generation results successfully!', '-'*10)
