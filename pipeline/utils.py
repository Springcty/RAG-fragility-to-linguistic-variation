import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

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

few_shot_examples = {
    'PopQA_Readability': [
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
    ],
    'PopQA_Formality': [
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
    ],
    'Entity_Readability': ...,
    'Entity_Formality': ...,
}

def create_few_shot_examples(args):
    few_shot_template = ChatPromptTemplate([
        ('human', 'Question: {input}'),
        ('ai', 'Answer: {output}')
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=few_shot_template,
        examples=few_shot_examples[f'{args.dataset}_{args.property}'],
    )
    return few_shot_prompt


def load_linguistic_query(args):
    load_path = f'{args.data_path}/{args.dataset}/{args.property}/metrics_filtered.csv'
    data = pd.read_csv(load_path)
    return data
    