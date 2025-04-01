import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


few_2shot_examples = {
    'popqa': {
        'readability': 'Question: What genre is Golden?\nAnswer: rock music\n\nQuestion: In which specific genre does the work titled "Golden" find its classification?\nAnswer: rock music',
        'politeness': 'Question: What genre is Golden?\nAnswer: rock music\n\nQuestion: Would you be so kind as to share with me what genre Golden falls under?\nAnswer: rock music',
        'formality': 'Question: What genre is Golden?\nAnswer: rock music\n\nQuestion: Hey, so like, do you know what genre Golden is?\nAnswer: rock music',
        'back_translated': 'Question: What genre is Golden?\nAnswer: rock music\n\nQuestion: What genre of Golden?\nAnswer: rock music',
        'edited_query_char': 'Question: What genre is Golden?\nAnswer: rock music\n\nQuestion: What genra is Golden?\nAnswer: rock music',
    },
    'entity_questions': {
        'readability': 'Question: Where was Michael Jack born?\nAnswer: Folkestone\n\nQuestion: In what geographical locale did the individual known as Michael Jackson enter into existence?\nAnswer: Folkestone',
        'politeness': 'Question: Where was Michael Jack born?\nAnswer: Folkestone\n\nQuestion: Would you be so kind as to share the birthplace of Michael Jack?\nAnswer: Folkestone',
        'formality': 'Question: Where was Michael Jack born?\nAnswer: Folkestone\n\nQuestion: Hey, so like, do you know where Michael Jack was born?\nAnswer: Folkestone',
        'back_translated': 'Question: Where was Michael Jack born?\nAnswer: Folkestone\n\nQuestion: Where was Michael Jacques born?\nAnswer: Folkestone',
        'edited_query_char': 'Question: Where was Michael Jack born?\nAnswer: Folkestone\n\nQuestion: Where wos Michael Jack born?\nAnswer: Folkestone',
    },
    'ms_marco': {
        'readability': 'Question: how long can chicken stay good in the fridge\nAnswer: 1 to 2 days\n\nQuestion: What is the time span within which chicken can sustain its quality for consumption when preserved in a refrigerated setting?\nAnswer: 1 to 2 days',
        'politeness': 'Question: how long can chicken stay good in the fridge\nAnswer: 1 to 2 days\n\nQuestion: Would you be so kind as to share how long chicken remains fresh in the refrigerator?\nAnswer: 1 to 2 days',
        'formality': 'Question: how long can chicken stay good in the fridge\nAnswer: 1 to 2 days\n\nQuestion: Hey, so like, do you know how long chicken can last in the fridge?\nAnswer: 1 to 2 days',
        'back_translated': 'Question: how long can chicken stay good in the fridge\nAnswer: 1 to 2 days\n\nQuestion: How long will chicken stay fresh in the refrigerator\nAnswer: 1 to 2 days',
        'edited_query_char': 'Question: how long can chicken stay good in the fridge\nAnswer: 1 to 2 days\n\nQuestion: how leng can chickon stay good in the fridge\nAnswer: 1 to 2 days',
    },
    'natural_questions': {
        'readability': 'Question: how many pieces in a terry\'s chocolate orange\nAnswer: six\n\nQuestion: What is the total quantity of individual segments contained within a Terry\'s chocolate orange confectionery item?\nAnswer: six',
        'politeness': 'Question: how many pieces in a terry\'s chocolate orange\nAnswer: six\n\nQuestion: Would you be so kind as to share the number of segments typically found in a Terry\'s chocolate orange?\nAnswer: six',
        'formality': 'Question: how many pieces in a terry\'s chocolate orange\nAnswer: six\n\nQuestion: Hey, so like, do you know a terry\'s chocolate orange contains how many pieces\nAnswer: six',
        'back_translated': 'Question: how many pieces in a terry\'s chocolate orange\nAnswer: six\n\nQuestion: How many pieces of Terry\'s Chocolate Orange\nAnswer: six',
        'edited_query_char': 'Question: how many pieces in a terry\'s chocolate orange\nAnswer: six\n\nQuestion: how meny pieces in a tarry\'s chocolate orange\nAnswer: six',
    },
}

readability_2_shot = '''
Question: What is Milton Bowen's occupation?
Answer: Teacher.

Question: In which urban locality did the individual known as Reham first emerge into existence?
Answer: Pittsburgh.
'''

formality_2_shot = '''
Question: What is Milton Bowen's occupation?
Answer: Teacher.

Question: Hey, so like, where was Reham born, like which city?
Answer: Pittsburgh.
'''


def create_few_shot_examples(args):
    few_shot_template = ChatPromptTemplate([
        ('human', 'Question: {input}'),
        ('ai', 'Answer: {output}')
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=few_shot_template,
        examples=few_2shot_examples[args.dataset][args.property],
    )
    return few_shot_prompt


def load_linguistic_query(args):
    load_path = f'{args.data_path}/{args.dataset}/{args.property}/metrics_filtered.csv'
    data = pd.read_csv(load_path)
    return data
    