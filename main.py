import os

from srcs.linear_regression_predictor import get_user_input
from srcs.train_model_class import premade_data
from srcs.test_linear_regression import TestLinearRegression


def main():
    """
    Main function to run the program.
    """
    print("Welcome to the car price predictor.")
    # Check if the model is already trained
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "json_files/model.pkl")
    if not os.path.exists(pickle_path):
        user_input = input("No model found. Would you like to train a new model? (y/n): ")
        if user_input.lower() == 'y':
            print("Training model. This may take a while.")
            premade_data()
        elif user_input.lower() == 'n':
            print("Exiting program.")
            return
        else:
            print("Invalid input. Exiting program.")
            return
    else:
        print("Model found.")

    # Run the tests
    user_input = input("Would you like to run the tests? (y/n): ")
    if user_input.lower() == 'y':
        test = TestLinearRegression()
        test.setUp()
        test.test_custom_model()
        test.test_error_handling()
    elif user_input.lower() == 'n':
        print("Let's predict some car prices.")
    else:
        print("Invalid input. Exiting program.")
        return
    get_user_input()


if __name__ == '__main__':
    main()
