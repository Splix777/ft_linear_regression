import csv
import numpy as np
import argparse


def generate_correlated_csv(file_name, feature1_name, feature2_name,
                            num_rows=1_00, correlation=0.7):
    # Generate random data for the first feature
    feature1 = np.random.rand(num_rows) * 1_000

    # Create a covariance matrix for the two features
    cov_matrix = np.array([[1, correlation], [correlation, 1]])

    # Generate correlated data using the covariance matrix
    correlated_data = np.random.multivariate_normal(
        [0, 0],
        cov_matrix,
        num_rows)

    feature2 = (feature1
                * correlation
                + correlated_data[:, 1]
                * 1_000
                * (1 - correlation**2)**0.5
                )

    # Create the CSV file and write the data
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow([feature1_name, feature2_name])
        # Write the data rows
        for f1, f2 in zip(feature1, feature2):
            csvwriter.writerow([f1, f2])

    print(f"CSV file '{file_name}' with {num_rows} rows of correlated features"
          f" '{feature1_name}' and '{feature2_name}' created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with two correlated features.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the output CSV file.")
    parser.add_argument(
        "--f1",
        type=str,
        required=True,
        help="The name of the first feature.")
    parser.add_argument(
        "--f2",
        type=str,
        required=True,
        help="The name of the second feature.")
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1_00,
        help="The number of rows in the CSV file. Default is 10000.")
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.9,
        help="The correlation coefficient between the features. Default 0.9.")

    args = parser.parse_args()

    generate_correlated_csv(
        args.name,
        args.f1,
        args.f2,
        args.num_rows,
        args.correlation
    )
