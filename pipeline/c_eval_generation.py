import os
import re
import argparse
import json
import ast

import pandas as pd
import numpy as np
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from joblib import Parallel, delayed

import warnings 
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--result_path', type=str, default='/data/group_data/maartens_lab_miis24/QL_result/gpt-4o-mini',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--retrieval', type=str, default='ModernBERT', 
                    help='The retrieval method from ["ModernBERT", "Contriever", "none_retrieval"]')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                    help='LLM used for generation')
parser.add_argument('--dataset', type=str, default='popqa',
                    help='Name of the QA dataset from ["popqa", "entity"]')
parser.add_argument('--linguistics', type=str, default='formality',
                    help='The linguistic properties of the query to be modified')
parser.add_argument('--modified', type=str, default='original',
                    help='The type of query to be modified, from ["original", "modified"]')

args = parser.parse_args()
print(args)


def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(pred, gold_answers):
    return int(normalize_answer(pred) in [normalize_answer(g) for g in gold_answers])


def check_included(pred, gold_answers):
    for g in gold_answers:
        if normalize_answer(g) in normalize_answer(pred):
            return 1
    return 0


def f1_score(pred, gold_answers):
    """Calculate F1 Score"""
    pred_tokens = normalize_answer(pred).split()
    f1_scores = []
    for answer in gold_answers:
        answer_tokens = normalize_answer(answer).split()
        common = set(pred_tokens) & set(answer_tokens)
        num_common = len(common)
        
        if num_common == 0:
            f1_scores.append(0)
            continue
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(answer_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return max(f1_scores)


def batch_bert_score(predictions, gold_answers_list):
    refs = [max(gold_answers, key=lambda g: len(g)) for gold_answers in gold_answers_list]  # Choose one ref per batch
    _, _, bert_scores = bert_score(predictions, refs, lang='en', rescale_with_baseline=True, device='cuda')
    return bert_scores


def rouge_score_fn(pred, gold_answers):
    """Calculate ROUGE Score"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(normalize_answer(ans), normalize_answer(pred)) for ans in gold_answers]
    max_scores = {
        key[:-1].upper()+'-'+key[-1].upper(): max(score[key].fmeasure for score in scores) for key in ['rouge1', 'rouge2', 'rougeL']
    }
    return max_scores


def transform_answers(cell):
    if isinstance(cell, list):
        return cell
    # Step 1: Convert the outer string to a list using ast.literal_eval
    list_as_string = ast.literal_eval(cell)
    return list_as_string


def eval_generation(df, save_path):
    # Load the generations
    df["answers"] = df["answers"].apply(transform_answers)
    gold_answers_list = df['answers'].tolist()
    predictions = df['generation'].tolist()
    # Truncate the generation and treat the first paragraph (split with ‘\n\n’) as the generated answer
    # predictions = [output['outputs'][0]['text'].split('\n')[0] for output in df['generation']]
    
    # Parallel exact match
    print('Parallel exact match evaluation...')
    em_scores = Parallel(n_jobs=-1)(delayed(em_score)(pred, gold) for pred, gold in zip(predictions, gold_answers_list))
    
    # Parallel check whether the gold answer is included in the model generation
    print('Parallel check whether the gold answer is included in the model generation...')
    ai_scores = Parallel(n_jobs=-1)(delayed(check_included)(pred, gold) for pred, gold in zip(predictions, gold_answers_list))

    # Parallel F1 Score
    print('Parallel F1 Score evaluation...')
    f1_scores = Parallel(n_jobs=-1)(delayed(f1_score)(pred, gold) for pred, gold in zip(predictions, gold_answers_list))

    # Batch BERTScore
    print('Batch BERTScore evaluation...')
    bert_scores = batch_bert_score(predictions, gold_answers_list)

    # ROUGE
    print('ROUGE evaluation...')
    rouge_scores = Parallel(n_jobs=-1)(
        delayed(lambda pred, gold: rouge_score_fn(pred, gold))(pred, gold)
        for pred, gold in zip(predictions, gold_answers_list)
    )
    rouge_1 = [score['ROUGE-1'] for score in rouge_scores]
    rouge_2 = [score['ROUGE-2'] for score in rouge_scores]
    rouge_l = [score['ROUGE-L'] for score in rouge_scores]
    
    scores_dict = {
        'AI': ai_scores,
        'EM': em_scores,
        'F1': f1_scores,
        'BERTScore': bert_scores,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
    }
    # Long-form result
    scores_df = pd.DataFrame(scores_dict)
    scores_df.to_csv(save_path)
    
    # Average result
    scores_avg_df = pd.DataFrame(scores_dict).mean()
    scores_avg_df.to_csv(save_path.replace('.csv', '_avg.csv'))
    print('Evaluation Finished!')


def eval_quantiles_popularity(df, save_path):
    quantiles = np.quantile(df['s_pop'], [0.25, 0.5, 0.75])
    df['quantile'] = pd.cut(
        df['s_pop'], 
        bins=[-np.inf, quantiles[0], quantiles[1], quantiles[2], np.inf], 
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
    )
    
    # Evaluate the generation for each quantile
    pop_scores = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        quantile_df = df[df['quantile'] == q]
        s_pop_list = quantile_df['s_pop'].tolist()
        pop_scores.append(sum(s_pop_list) / len(s_pop_list))
        eval_generation(quantile_df, f'{save_path}/{args.modified}_{q}_longform.csv')
    
    return pop_scores


def main():
    # /data/group_data/maartens_lab_miis24/QL_result/gpt-4o-mini/entity_questions/politeness/ModernBERT/Llama-3.1-8B-Instruct/modified_generation.jsonl
    data_path = os.path.join(args.result_path, args.dataset, args.linguistics, args.retrieval, args.model_name.split('/')[1])
    save_path = f'{data_path}/{args.modified}_generation_score.csv'
    
    if os.path.exists(save_path):
        print(f'The {save_path} evaluation file already exists!')
        return
    
    print('Loading the generation data from:', f'{data_path}/{args.modified}_generation.jsonl')
    df = pd.read_json(f'{data_path}/{args.modified}_generation.jsonl', orient='records', lines=True)
    
    # # HotPotQA evaluation
    # root_dir = '/data/group_data/maartens_lab_miis24/MultiHop'
    # data_path = os.path.join(root_dir, args.linguistics, f'{args.modified}_generation_qsampled.jsonl')
    # save_path = f'{root_dir}/{args.linguistics}/{args.modified}_generation_qsampled_score.csv'
    # df = pd.read_json(data_path, orient='records', lines=True)
    
    print('Begin evaluation...')
    eval_generation(df, save_path)
    
    # pop_scores = eval_quantiles_popularity(df, res_path)
    # print(args)
    # print(pop_scores)
    # print('-'*10)
    

if __name__ == '__main__':
    main()