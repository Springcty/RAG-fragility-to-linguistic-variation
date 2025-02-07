import re
import nltk
from nltk import pos_tag, word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

concreteness_df = pd.read_excel('/home/neelbhan/QueryLinguistic/query_rewriting/Concreteness_ratings_Brysbaert_et_al_BRM(1).xlsx')
word_concreteness = dict(zip(concreteness_df['Word'], concreteness_df['Conc.M']))

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

def main():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('--data', type=str, default='data/processed_data.csv',
                        help='The path to load the dataset')
    parser.add_argument('--output', type=str, default='data/scored_data.csv',
                        help='The path to save the scored dataset')
    parser.add_argument('--output_modified', type=str, default='data/scored_data.csv',
                        help='The path to save the scored dataset')
    args = parser.parse_args()
    final_header = pd.read_csv(args.data)

    model_name = "s-nlp/roberta-base-formality-ranker"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    formal_bert=[]
    concreteness=[]
    formality_score=[]
    readability=[]
    if 'original_query' not in final_header.columns:
        tag= 'Original Query'
    else:
        tag= 'original_query'
    for i in tqdm(range(len(final_header)), desc="Processing headers"):
        text = final_header[tag][i]
        formal_bert.append(predict_formality(tokenizer=tokenizer, model=model, text=text))
        concreteness.append(sentence_concreteness(text))
        formality_score.append(calculate_formality(text))
        readability.append(flesch_reading_ease(text))


    df = pd.DataFrame({
        "question": final_header[tag],
        "roberta_formality_score": formal_bert,
        "concreteness_score": concreteness,
        "linguistic_formality_score": formality_score,
        "readability": readability,
        'possible_answers': final_header['possible_answers'],
    })

    df.to_csv(args.output, index=False)
    #Return final header without question column and rename modified_query to question
    final_header = final_header.drop(columns=[tag])
    final_header.rename(columns={'modified_query': 'question'}, inplace=True)
    final_header.to_csv(args.output_modified, index=False)


if __name__ == "__main__":
    main()