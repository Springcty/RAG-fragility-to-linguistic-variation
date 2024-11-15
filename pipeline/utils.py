import pandas as pd
import ast

def load_linguistic_query(args):
    load_path = f'{args.data_path}/{args.dataset}/{args.property}'
    
    df_questions = pd.read_csv(f'{load_path}/queries.csv')
    questions_o = df_questions['Original Question'].tolist()
    questions_m = df_questions['Modified Question'].tolist()
    scores_o = df_questions['Original Score'].tolist()
    scores_m = df_questions['Modified Score'].tolist()
    
    df_answers = pd.read_csv(f'{load_path}/answers.csv')
    answers = df_answers['answers'].tolist()
    answers = [ast.literal_eval(answer) for answer in answers]
    
    return questions_o, questions_m, scores_o, scores_m, answers