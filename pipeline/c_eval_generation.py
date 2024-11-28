import pandas as pd
from joblib import Parallel, delayed
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import re
import argparse
import math

parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/user_data/tianyuca',
                    help='The path to load the dataset')
parser.add_argument('--result_path', type=str, default='/data/user_data/tianyuca/QL_result',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--dataset', type=str, default='popqa',
                    help='Name of the QA dataset from ["popqa", "entity"]')
parser.add_argument('--property', type=str, default='readability',
                    help='The linguistic properties of the query to be modified')
parser.add_argument('--modified', type=str, default='modified')
args = parser.parse_args()

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


def exact_match_vectorized(predictions, gold_answers_list):
    def check_em(pred, gold_answers):
        return int(normalize_answer(pred) in [normalize_answer(g) for g in gold_answers])
    return [check_em(pred, gold) for pred, gold in zip(predictions, gold_answers_list)]


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


def optimized_evaluate_qa(data_path, res_path, filtered_dataset_path):
    # Merge dataframes
    # Load the dataset and the generations
    dataset = pd.read_csv(f'{data_path}/metrics_filtered.csv')
    dataset_filtered = pd.read_csv(filtered_dataset_path)
    generations = pd.read_csv(f'{res_path}/retrieval_generations.csv')
    
    # print(generations['modified_retrieval_ids'])
    print(len(generations))
    generations = generations[generations['original_retrieval_ids'].apply(eval).apply(lambda x: len(x) > 0)]
    # generations = generations[generations['modified_retrieval_ids'].apply(eval).apply(lambda x: len(x) > 0)]
    print(len(generations))
    
    # Perform a shitty merge
    merged_dataset = pd.merge(
        dataset[['original_query', 'modified_query', 'question_id']],
        dataset_filtered[['original_query', 'possible_answers']],
        on='original_query',
        how='inner'
    )
    
    merged_df = pd.merge(
        merged_dataset[['question_id', 'possible_answers']],
        generations[['question_id', 'original_answer', 'modified_answer']],
        on='question_id',
        how='inner'
    )

    # Extract predictions and gold answers
    predictions = merged_df['original_answer'].tolist()
    # predictions = merged_df['modified_answer'].tolist()
    gold_answers_list = merged_df['possible_answers'].apply(eval).tolist()  # Assuming gold_answers is a stringified list

    # Parallel exact match
    em_scores = exact_match_vectorized(predictions, gold_answers_list)

    # Parallel F1 Score
    f1_scores = Parallel(n_jobs=-1)(delayed(f1_score)(pred, gold) for pred, gold in zip(predictions, gold_answers_list))

    # Batch BERTScore
    bert_scores = batch_bert_score(predictions, gold_answers_list)

    # ROUGE
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = Parallel(n_jobs=-1)(
        delayed(lambda pred, gold: rouge_score_fn(pred, gold))(pred, gold)
        for pred, gold in zip(predictions, gold_answers_list)
    )
    rouge_1 = [score['ROUGE-1'] for score in rouge_scores]
    rouge_2 = [score['ROUGE-2'] for score in rouge_scores]
    rouge_l = [score['ROUGE-L'] for score in rouge_scores]

    # Add results to DataFrame
    merged_df['EM'] = em_scores
    merged_df['F1'] = f1_scores
    merged_df['BERTScore'] = bert_scores
    merged_df['ROUGE-1'] = rouge_1
    merged_df['ROUGE-2'] = rouge_2
    merged_df['ROUGE-L'] = rouge_l

    # merged_df.to_csv('evaluation.csv')

    average_scores = merged_df[['EM', 'F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']].mean()
    average_scores.to_csv('read_original_score_filtered.csv')
    # average_scores.to_csv('read_modified_score_filtered.csv')

    
    return merged_df


def eval_generation(res_path):
    # # Load the dataset and the generation results
    # dataset = pd.read_csv(f'{data_path}/metrics_filtered.csv')
    # generations = pd.read_csv(f'{res_path}/retrieval_generations.csv')
    
    # # Merge generated answers with gold answers according to the question_id
    # merged_df = pd.merge(
    #     dataset[['question_id', 'possible_answers']],
    #     generations[['question_id', 'original_answer', 'modified_answer']],
    #     on='question_id',
    #     how='inner',
    # )
    
    # # Extract generations and gold answers
    # predictions = merged_df[f'{query_type}_answer'].tolist()
    # gold_answers_list = merged_df['possible_answers'].apply(eval).tolist()  # possible_answewrs is a stringified list

    # Load the generations
    data_df = pd.read_json(f'{res_path}/{args.modified}_generation.jsonl', orient='records', lines=True)
    # predictions = [output['outputs'][0]['text'].split('\n')[0] for output in data_df['generation']]
    predictions = [output['outputs'][0]['text'] for output in data_df['generation']]
    gold_answers_list = data_df['answers'].tolist()

    # Parallel exact match
    em_scores = exact_match_vectorized(predictions, gold_answers_list)

    # Parallel F1 Score
    f1_scores = Parallel(n_jobs=-1)(delayed(f1_score)(pred, gold) for pred, gold in zip(predictions, gold_answers_list))

    # Batch BERTScore
    bert_scores = batch_bert_score(predictions, gold_answers_list)

    # ROUGE
    rouge_scores = Parallel(n_jobs=-1)(
        delayed(lambda pred, gold: rouge_score_fn(pred, gold))(pred, gold)
        for pred, gold in zip(predictions, gold_answers_list)
    )
    rouge_1 = [score['ROUGE-1'] for score in rouge_scores]
    rouge_2 = [score['ROUGE-2'] for score in rouge_scores]
    rouge_l = [score['ROUGE-L'] for score in rouge_scores]
    
    scores_dict = {
        'EM': em_scores,
        'F1': f1_scores,
        'BERTScore': bert_scores,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
    }
    scores_df = pd.DataFrame(scores_dict).mean()
    scores_df.to_csv(f'{res_path}/{args.modified}_generation(all)_score.csv')
    print('Evaluation Finished!')
    

res_path = f'{args.result_path}/{args.dataset}/{args.property}'
eval_generation(res_path)