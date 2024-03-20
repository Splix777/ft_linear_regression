from srcs.train import train
from srcs.predict_price import predict
from srcs.visualize_data import visualize_data
from srcs.calculate_precision import precision_of_algorithm


if __name__ == '__main__':
    print("Welcome to ft_linear_regression!")
    print("1. Train the model")
    print("2. Predict")
    print("3. Visualize Data")
    print("4. Calculate Precision of the Algorithm")
    print("5. Exit")
    option = input("Please, choose an option: ")
    if option == "1":
        print("Training the model...")
        train()
    elif option == "2":
        print("Predicting...")
        predict()
    elif option == "3":
        print("Visualizing data...")
        visualize_data()
    elif option == "4":
        print("Calculating precision...")
        precision_of_algorithm()
    elif option == "5":
        print("Exiting...")
        exit()
    else:
        print("Invalid option")
        exit()
