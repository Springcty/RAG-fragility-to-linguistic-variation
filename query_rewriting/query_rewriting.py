import re
import nltk
from nltk import pos_tag, word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import argparse
import asyncio
import logging
import os
from datasets import load_dataset
import random
import aiolimiter
import openai
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

print("API KEY",os.getenv("OPENAI_API_KEY"))
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

concreteness_df = pd.read_excel('/Users/neelbhandari/QueryLinguistic/query_rewriting/Concreteness_ratings_Brysbaert_et_al_BRM(1).xlsx')
word_concreteness = dict(zip(concreteness_df['Word'], concreteness_df['Conc.M']))

ERROR_ERRORS_TO_MESSAGES = {
    openai.AuthenticationError: "Authentication Error: {e}. Usually due to incorrect OPENAI_API_KEY.",
    openai.BadRequestError: "Bad Request Error: {e}. Usually due to missing or invalid parameters.",
    openai.PermissionDeniedError: "Permission Denied Error: {e}. You don't have access to the requested resource.",
    openai.NotFoundError: "Not Found Error: {e}. The requested resource doesn't exist.",
    openai.APIConnectionError: "API Connection Error: {e}. There was a problem connecting to the OpenAI API.",
    openai.APITimeoutError: "API Timeout Error: {e}. The request timed out.",
    openai.InternalServerError: "Internal Server Error: {e}. An error occurred on the server side.",
    openai.RateLimitError: "Rate Limit Error: {e}. You've hit the OpenAI API rate limit.",
    openai.UnprocessableEntityError: "Unprocessable Entity Error: {e}. Unable to process the request despite the format being correct.",
}


def count_syllables(word):
    """Counts syllables in a word based on the number of vowel groups."""
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syllable_count -= 1
    return max(1, syllable_count)

def flesch_reading_ease(text):
    """Calculates the Flesch Reading Ease Score (FRES) for the given text."""
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s]
    words = re.findall(r'\w+', text)

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)

    if num_sentences == 0 or num_words == 0:
        return None  

    avg_words_per_sentence = num_words / num_sentences
    avg_syllables_per_word = num_syllables / num_words

    fres = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    return fres

def calculate_formality(sentence):
    # Tokenize the sentence and get POS tags
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)

    # Initialize counts for each POS category
    noun_count = adjective_count = preposition_count = article_count = 0
    pronoun_count = verb_count = adverb_count = interjection_count = 0

    # Define tags for each category based on Penn Treebank POS tags
    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
    adjective_tags = {"JJ", "JJR", "JJS"}
    preposition_tags = {"IN"}
    article_tags = {"DT"}
    pronoun_tags = {"PRP", "PRP$", "WP", "WP$"}
    verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    adverb_tags = {"RB", "RBR", "RBS"}
    interjection_tags = {"UH"}

    # Count the occurrences of each part of speech
    for word, tag in pos_tags:
        if tag in noun_tags:
            noun_count += 1
        elif tag in adjective_tags:
            adjective_count += 1
        elif tag in preposition_tags:
            preposition_count += 1
        elif tag in article_tags:
            article_count += 1
        elif tag in pronoun_tags:
            pronoun_count += 1
        elif tag in verb_tags:
            verb_count += 1
        elif tag in adverb_tags:
            adverb_count += 1
        elif tag in interjection_tags:
            interjection_count += 1

    
    total_words = len(words)
    if total_words == 0:
        return None  


    noun_freq = noun_count / total_words * 100
    adjective_freq = adjective_count / total_words * 100
    preposition_freq = preposition_count / total_words * 100
    article_freq = article_count / total_words * 100
    pronoun_freq = pronoun_count / total_words * 100
    verb_freq = verb_count / total_words * 100
    adverb_freq = adverb_count / total_words * 100
    interjection_freq = interjection_count / total_words * 100


    formality_score = (noun_freq + adjective_freq + preposition_freq + article_freq
                       - pronoun_freq - verb_freq - adverb_freq - interjection_freq + 100) / 2
    return formality_score

def predict_formality(tokenizer,model,text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    formal_probability = probabilities[0][1].item()

    return formal_probability

def sentence_concreteness(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence.lower())

    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)

    # Filter out non-content words
    content_words = [word for word, pos in pos_tags if pos.startswith(('N', 'V', 'J', 'R'))]

    # Get concreteness scores for content words
    concreteness_scores = [word_concreteness.get(word, 0) for word in content_words]

    # Calculate average concreteness
    if concreteness_scores:
        return np.mean(concreteness_scores)
    else:
        return 0

def format_prompt(prompt_input):
    prompt = prompt_input.split("\t")[-1]
    return [{"role": "user", "content": prompt}]


async def _throttled_openai_chat_completion_acreate(
    model,
    messages,
    temperature,
    max_tokens,
    top_p,
    limiter: aiolimiter.AsyncLimiter,
    num_responses_per_prompt,
):
    async with limiter:
        for _ in range(3):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=num_responses_per_prompt,
                )
            except (
                openai.AuthenticationError,
                openai.BadRequestError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
            except openai.UnprocessableEntityError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(10)
            except openai.RateLimitError as e:
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(30)
            except Exception as e:
                logging.warning(e)
            await asyncio.sleep(30)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
    requests_per_minute,
    num_responses_per_prompt,
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        prompts: Prompts.
        model: Model.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.
        num_responses_per_prompt: Number of responses to generate per prompt.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            num_responses_per_prompt=num_responses_per_prompt,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return responses


def main(
    prompts,
    model,
    temperature,
    max_tokens,
    top_p,
    requests_per_minute,
    num_responses_per_prompt,
):
    messages_list = [format_prompt(prompt) for prompt in prompts]
    predictions = asyncio.run(
        generate_from_openai_chat_completion(
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            requests_per_minute=requests_per_minute,
            num_responses_per_prompt=num_responses_per_prompt,
        )
    )
    results = []
    for prompt, prediction in zip(prompts, predictions):
        prompt = prompt.split("\t")
        for x in range(num_responses_per_prompt):
            if x >= len(prediction.choices):
                prompt.append("")
                continue
            prompt.append(
                prediction.choices[x]
                .message.content.replace("\n", "~| ")
                .replace("\t", " ")
            )
        # prompt.append(
        #     prediction.choices[0]
        #     .message.content.replace("\n", "~| ")
        #     .replace("\t", " ")
        # )
        results.append(tuple(prompt))
    return results

def randomize(data, num_samples, use_full=False):
    random.seed(42)

    print(use_full)
    print(f"Number of examples: {len(train_dataset)}")
    print(f"Column names: {train_dataset.column_names}")
    if use_full:
        random_indices = range(len(train_dataset))
    else:
        random_indices = random.sample(range(len(train_dataset)), 20)

    samples= []
    ans=[]
    print(len(random_indices))
    # View a few examples
    for i in random_indices:
        # print(f"\nExample {i+1}:")
        # print(train_dataset[i]['question'])
        samples.append(train_dataset[i]['question'])
        # print(f"Subject: {train_dataset[i]['subj']}")
        # print(f"Object: {train_dataset[i]['obj']}")
        # print(f"Relationship: {train_dataset[i]['prop']}")
        # print(f"Possible Answers: {train_dataset[i]['possible_answers']}")
        ans.append(train_dataset[i]['possible_answers'])
    # print(len(samples))
    return samples

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--output_og",
        type=str,
    )
    argparser.add_argument(
        "--output_mod",
        type=str,
    )

    argparser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    argparser.add_argument("--temperature", type=float, default=0.7)
    argparser.add_argument("--max_response_tokens", type=int, default=100)
    argparser.add_argument("--top_p", type=float, default=1.0)  # Don't alter this
    argparser.add_argument("--requests_per_minute", type=int, default=150)
    argparser.add_argument("--num_responses_per_prompt", type=int, default=1)
    argparser.add_argument("--full_sampling", type=bool, default=False)
    argparser.add_argument("--modification", type=str, default="formality")
    argparser.add_argument("--dataset", type=str, default="akariasai/PopQA")
    
    args = argparser.parse_args()

    if args.dataset == "PopQA":
        dataset = load_dataset("akariasai/PopQA")
        print("Dataset loaded")
        train_dataset = dataset['test']
        samples=20
        print(args.full_sampling)
        random_samples= randomize(train_dataset, samples, bool(args.full_sampling))
    
    elif args.dataset == "EntityQuestion":
        pass

    elif args.dataset == "TriviaQA":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc",split="test")
        print("Dataset loaded")
        train_dataset = dataset['test']
        samples=20
        print(args.full_sampling)
        random_samples= randomize(train_dataset, samples, bool(args.full_sampling))
    
    


    # with open(args.prompts) as f:
    #     prompts = f.readlines()
    if args.modification == "formality":
        formal_prompt_prefix= """You are a helpful query rewriting AI assistant. Your task is to convert the given retrieval knowledge query into a very informal and casual query that can have an impact on retrieval. 
Use your lingustic knowledge to convert the query to an informal query. Examples of techniques that can be used include changing Conversational Tone, using filler words, slangs and idioms, Misspellings and Typos among other things. You must always preserve the intent of the query. Original Query:"""
        prompts = [formal_prompt_prefix + query + ". Informal Query: " for query in random_samples]
    else:
        read_prompt_prefix = """You are a helpful query rewriting AI assistant. Your task is to convert the given retrieval knowledge query into a query with low readaility that can have an impact on retrieval. 
Use your lingustic knowledge to convert the query to a low readability query. Examples of techniques that can be used include increasing lexical and syntactic complexity, semantic ambiguity, among other things. You must always preserve the intent of the query. Original Query:"""
    # prompts = [formal_prompt_prefix + query + ". Informal Query: " for query in random_samples]
        prompts = [read_prompt_prefix + query + ". Query with Low Readability: " for query in random_samples]
    
    results = main(
        prompts,
        args.model,
        args.temperature,
        args.max_response_tokens,
        args.top_p,
        args.requests_per_minute,
        args.num_responses_per_prompt,
    )

    final_header= pd.DataFrame(results, columns=['prompt', 'response'])
    final_header['query']= random_samples

    final_header.to_csv(args.output, index=None)

    model_name = "s-nlp/roberta-base-formality-ranker"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    formal_bert=[]
    concreteness=[]
    formality_score=[]
    readability= []
    formal_bert_q=[]
    concreteness_q=[]
    formality_score_q=[]
    readability_q= []
    for i in range(len(final_header)):
        text= final_header['response'][i]
        formal_bert.append(predict_formality(tokenizer=tokenizer,model=model,text=text))
        concreteness.append(sentence_concreteness(text))
        formality_score.append(calculate_formality(text))
        readability.append(flesch_reading_ease(text))

        query_text= final_header['query'][i]
        formal_bert_q.append(predict_formality(tokenizer=tokenizer,model=model,text=query_text))
        concreteness_q.append(sentence_concreteness(query_text))
        formality_score_q.append(calculate_formality(query_text))
        readability_q.append(flesch_reading_ease(query_text))

    df_q= pd.DataFrame({
        "question": final_header['query'],
        "Roberta Formality Ranking Score": formal_bert_q,
        "Concreteness Score": concreteness_q,
        "Linguistic Formality Score": formality_score_q,
        "Readability": readability_q
    })

    df_q.to_csv(args.output_og, index=None)


    df = pd.DataFrame({
        # "question": final_header['query'],
        "question": final_header['response'],
        "Roberta Formality Ranking Score": formal_bert,
        "Concreteness Score": concreteness,
        "Linguistic Formality Score": formality_score,
        "Readability": readability
    })

    df.to_csv(args.output_mod, index=None)

    
#What is the terminal command to run this file with TriviaQA dataset and formality modification
#python query_rewriting/query_rewriting.py --output_og sample_testing/output_og.csv --output_mod sample_testing/output_mod.csv --dataset TriviaQA --modification formality --full_sampling False



