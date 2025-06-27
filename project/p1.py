# square_game.py

import datetime
from exceptions import get_valid_name, get_valid_int_input, get_yes_or_no
from Questions import get_random_question

def square_game():
    name = get_valid_name()
    game_start_time = datetime.datetime.now()

    print(f"Hello, {name}! Welcome to the Timed Square & Root Game ")
    print(f" Game started at: {game_start_time.strftime('%H:%M:%S')}\n")

    score = 0

    while True:
        # Get random question and answer from module
        question_text, correct_answer = get_random_question()
        print(f"\n {question_text}")

        question_start = datetime.datetime.now()
        user_answer = get_valid_int_input("Your answer: ")
        question_end = datetime.datetime.now()

        time_taken = question_end - question_start

        if user_answer == correct_answer:
            score += 10
            print(f" Correct! +10 points ( Took {time_taken.total_seconds():.2f} seconds)")
        else:
            score -= 5
            print(f" Wrong! The correct answer was {correct_answer}. -5 points (⏱ Took {time_taken.total_seconds():.2f} seconds)")

        print(f" Current Score: {score}")

        play_more = get_yes_or_no("Do you want to play another question? (yes/no): ")
        if play_more != "yes":
            break

    game_end_time = datetime.datetime.now()
    total_time = game_end_time - game_start_time

    print(f"\nThanks for playing, {name}!  Your final score is {score}.")
    print(f" Game ended at: {game_end_time.strftime('%H:%M:%S')}")
    print(f" Total play time: {total_time}")

# Run the game
if __name__ == "__main__":
    square_game()
