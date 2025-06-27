# exceptions.py

def get_valid_name():
    while True:
        try:
            name = input("What is your name? ").strip()
            if not name:
                raise ValueError("Name cannot be empty.")
            return name
        except ValueError as e:
            print(f" Error: {e}")
        except KeyboardInterrupt:
            print("\n Program exited by user.")
            exit()

def get_valid_int_input(prompt):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                raise ValueError("Input cannot be empty.")
            return int(user_input)
        except ValueError:
            print(" Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n Game interrupted by user.")
            exit()

def get_yes_or_no(prompt):
    while True:
        try:
            response = input(prompt).strip().lower()
            if response not in ['yes', 'no']:
                raise ValueError("Please type 'yes' or 'no'.")
            return response
        except ValueError as e:
            print(f" {e}")
        except KeyboardInterrupt:
            print("\n Game interrupted by user.")
            exit()
