import argparse
from tqdm import tqdm
import json
import os
import re
# Set up LangchainSmith for trace
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = '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'
from langchain_core.tracers.context import tracing_v2_enabled

import torch
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_community.retrievers import WikipediaRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils import load_linguistic_query


parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/user_data/tianyuca',
                    help='The path to load the dataset')
parser.add_argument('--result_path', type=str, default='/data/user_data/tianyuca/QL_result',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--dataset', type=str, default='PopQA',
                    help='Name of the QA dataset from ["PopQA", "EntityQuestions"]')
parser.add_argument('--property', type=str, default='Readability',
                    help='The linguistic properties of the query to be modified, from["Readability", "Formality", "Concreteness"]')
parser.add_argument('--embedding_model_name', type=str, default='nvidia/NV-Embed-v2',
                    help='Embedding model used for generating embeddings')
parser.add_argument('--customize_retriever', action='store_true',
                    help='True if use customized retriever instead of WikipediaRetriever from LangChain')
parser.add_argument('--model_name', type=str, default='/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct',
                    help='LLM used for generation')

args = parser.parse_args()


def main():
    # Load the dataset
    dataset = load_linguistic_query(args)[:10]

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
        model_kwargs={'torch_dtype': torch.bfloat16, 'temperature': 0},
        max_new_tokens=128,
        device='cuda',
        return_full_text=False, # avoids including the input prompt in the output
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
    few_shot_PopQA_readability = [
        {
            "input": "What is Milton Bowen's occupation?",
            "output": "Teacher."
        },{
            "input": "In which urban locality did the individual known as Reham first emerge into existence?",
            "output": "Pittsburgh."
        },{
            "input": "What genre is Alana?",
            "output": "Music."
        },{
            "input": "Who might be identified as the progenitor of the individual historically recognized as Jacobs?",
            "output": "Esa."
        },{
            "input": "In what country is Skyla?",
            "output": "Chase."
        },{
            "input": "Inquiring into the identity of the individual or entity responsible for the production of the cinematic work entitled 'Demoes'?",
            "output": "Meta."
        },{
            "input": "Who was the director of Ahmed?",
            "output": "Barrett."
        },{
            "input": "To which geopolitical entity does Luc serve as the administrative capital?",
            "output": "Caervantes."
        },{
            "input": "Who was the screenwriter for Randy?",
            "output": "Buckley."
        },{
            "input": "Inquiring into the identity of the individual responsible for the composition of the musical piece commonly referred to as 'Ava'?",
            "output": "Holder."
        },{
            "input": "What color is Leonie?",
            "output": "Yoder."
        },{
            "input": "What specific theological belief system is adhered to by individuals identifying as Jose?",
            "output": "Wise."
        },{
            "input": "What sport does Arran play?",
            "output": "Gutierrez."
        },{
            "input": "In the realm of literary authorship, which individual can be ascribed the creative genesis of the work entitled Damian?",
            "output": "Singh."
        },{
            "input": "Who is the mother of Jasmine?",
            "output": "Weeks."
        },{
            "input": "What singular urban jurisdiction can be designated as the principal administrative center of the geographic region known as Axel?",
            "output": "Petty."
        },
    ]
    few_shot_PopQA_formality = [
        {
            "input": "What is Milton Bowen's occupation?",
            "output": "Teacher."
        },{
            "input": "Hey, so like, where was Reham born, like which city?",
            "output": "Pittsburgh."
        },{
            "input": "What genre is Alana?",
            "output": "Music."
        },{
            "input": "Who's Jacob's dad?",
            "output": "Esa."
        },{
            "input": "In what country is Skyla?",
            "output": "Chase."
        },{
            "input": "Yo, who was the dude that produced Demoes?",
            "output": "Meta."
        },{
            "input": "Who was the director of Ahmed?",
            "output": "Barrett."
        },{
            "input": "So, like, what's Kueaf the capital of, anyway?",
            "output": "Caervantes."
        },{
            "input": "Who was the screenwriter for Randy?",
            "output": "Buckley."
        },{
            "input": "Yo, who made Ava?",
            "output": "Holder."
        },{
            "input": "What color is Leonie?",
            "output": "Yoder."
        },{
            "input": "So, like, what was Jose's religion, ya know?",
            "output": "Wise."
        },{
            "input": "What sport does Arran play?",
            "output": "Gutierrez."
        },{
            "input": "Yo, who wrote Damian?",
            "output": "Singh."
        },{
            "input": "Who is the mother of Jasmine?",
            "output": "Weeks."
        },{
            "input": "Hey, what's the capital of Axel, ya know?",
            "output": "Petty."
        },
    ]

    few_shot_template = ChatPromptTemplate([
        ('human', 'Question: {input}'),
        ('ai', 'Answer: {output}')
    ])
    
    if args.dataset == 'PopQA' and args.property == 'Readability':
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_template,
            examples=few_shot_PopQA_readability
        )
    elif args.dataset == 'PopQA' and args.property == 'Formality':
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_template,
            examples=few_shot_PopQA_formality
        )

    prompt = ChatPromptTemplate([
        ('system', 'You are a professional question-answer task assistant. Use the following pieces of retrieved context to answer the question briefly. \n\nContext: {context}\n\nBelow are examples of questions and answers:'),
        few_shot_prompt,
        ('human', 'Now, it\'s your turn to answer the question below. The answer should contain only one sentence.\n\nQuestion: {input}\n\nAnswer:')
    ])

    questions_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, questions_answer_chain)
    print('-'*10, 'RAG chain setup successfully!', '-'*10)
    print('-'*10, 'Begin query!', '-'*10)

    # Answer generation
    generations = []
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        query_id = row['question_id']
        original_query = row['original_query']
        modified_query = row['modified_query']
        # original_score = row['original_score']
        # modified_score = row['modified_score']
        # gold_answers = row['answer']
        
        with tracing_v2_enabled(project_name=f'{args.dataset}_{args.property}_fewshot'):
            o_generation = rag_chain.invoke({'input': original_query})
            m_generation = rag_chain.invoke({'input': modified_query})
        
        o_retrieval_ids, m_retrieval_ids = [], []
        o_retrieval_contents, m_retrieval_contents = [], []
        for doc in o_generation['context']:
            o_retrieval_ids.append(doc.metadata['source'])
            o_retrieval_contents.append(doc.page_content)
        for doc in m_generation['context']:
            m_retrieval_ids.append(doc.metadata['source'])
            m_retrieval_contents.append(doc.page_content)
        o_answer = o_generation['answer']
        m_answer = m_generation['answer']

        result = {
            'question_id': query_id,
            'original_retrieval_ids': o_retrieval_ids,
            'modified_retrieval_ids': m_retrieval_ids,
            'orginal_retrieval': o_retrieval_contents,
            'modified_retrieval': m_retrieval_contents,
            'original_answer': o_answer,
            'modified_answer': m_answer
        }
        generations.append(result)

    save_path = f'{args.result_path}/{args.dataset}/{args.property}'
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame(generations)
    df.to_csv(f'{save_path}/retrieval_generations.csv')
    print('-'*10, 'Save retrieval and generation results successfully!', '-'*10)


if __name__ == "__main__":
    main()