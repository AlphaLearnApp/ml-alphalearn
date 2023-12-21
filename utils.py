import pandas as pd
import random

def get_random_question(df):
    random_question_index = random.choice(df.index)
    random_question = df.loc[random_question_index, 'Question']
    choices = df.loc[random_question_index, 'Choices']
    correct_answer = df.loc[random_question_index, 'Correct Answer']

    return random_question, choices, correct_answer