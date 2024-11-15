import json
import argparse

parser = argparse.ArgumentParser(description='RAG pipeline')
parser.add_argument('--data_path', type=str, default='/data/user_data/tianyuca',
                    help='The path to load the dataset')
parser.add_argument('--result_path', type=str, default='/data/user_data/tianyuca/QL_result',
                    help='The path to save the retrievel and generation results')
parser.add_argument('--dataset', type=str, default='PopQA',
                    help='Name of the QA dataset from ["PopQA", "EntityQuestions"]')
parser.add_argument('--property', type=str, default='Readability',
                    help='The linguistic properties of the query to be modified')

args = parser.parse_args()

def eval_retrieval_property(args):
    load_path = f'{args.result_path}/{args.dataset}/{args.property}'
    rag_dict_o = json.load(open(f'{load_path}/results_original_query.json', 'r'))
    rag_dict_m = json.load(open(f'{load_path}/results_modified_query.json', 'r'))
    
    retrieval_o = list(rag_dict_o.values())
    retrieval_m = list(rag_dict_m.values())
    match_cnt = 0
    total_cnt_o = 0
    
    total_score_o = 0
    total_score_m = 0
    
    for i in range(len(retrieval_o)):
        doc_ids_o = retrieval_o[i]['retrieval_doc_id']
        doc_ids_m = retrieval_m[i]['retrieval_doc_id']
        for id in doc_ids_m:
            if id in doc_ids_o:
                match_cnt += 1
        total_cnt_o += len(doc_ids_o)
        total_score_o += retrieval_o[i]['score']
        total_score_m += retrieval_m[i]['score']
    
    return match_cnt / total_cnt_o, total_score_o / len(retrieval_o), total_score_m / len(retrieval_m)

match_ratio, score_o, score_m = eval_retrieval_property(args)
print(match_ratio, score_o, score_m)