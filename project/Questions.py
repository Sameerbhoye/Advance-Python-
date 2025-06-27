# questions.py

import random
import math

def generate_square_question():
    number = random.randint(1, 20)
    question_text = f"What is the square of {number}?"
    correct_answer = number ** 2
    return question_text, correct_answer

def generate_square_root_question():
    perfect_squares = [i**2 for i in range(1, 11)]  # 1^2 to 10^2
    number = random.choice(perfect_squares)
    question_text = f"What is the square root of {number}?"
    correct_answer = int(math.sqrt(number))
    return question_text, correct_answer

def get_random_question():
    question_type = random.choice(['square', 'squareroot'])
    if question_type == 'square':
        return generate_square_question()
    else:
        return generate_square_root_question()
