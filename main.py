from srcs.train import train
from srcs.predict_price import predict


if __name__ == '__main__':
    print("Welcome to ft_linear_regression!")
    print("1. Train the model")
    print("2. Predict")
    print("3. Exit")
    option = input("Please, choose an option: ")
    if option == "1":
        print("Training the model...")
        train()
    elif option == "2":
        print("Predicting...")
        predict()
    elif option == "3":
        print("Exiting...")
        exit()
    else:
        print("Invalid option")
        exit()
