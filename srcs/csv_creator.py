import csv
import numpy as np
import argparse


def generate_correlated_csv(file_name, feature1_name, feature2_name, num_rows=10000, correlation=0.9):
    # Generate random data for the first feature
    feature1 = np.random.rand(num_rows) * 1_000_000

    # Generate the second feature with some correlation to the first feature
    noise = np.random.randn(num_rows) * (1 - correlation)
    feature2 = feature1 * correlation + noise * 1_000_000

    # Create the CSV file and write the data
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow([feature1_name, feature2_name])
        # Write the data rows
        for f1, f2 in zip(feature1, feature2):
            csvwriter.writerow([f1, f2])

    print(f"CSV file '{file_name}' with {num_rows} rows of correlated features '{feature1_name}' and '{feature2_name}' created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV file with two correlated features.")
    parser.add_argument("file_name", type=str, help="The name of the output CSV file.")
    parser.add_argument("feature1_name", type=str, help="The name of the first feature.")
    parser.add_argument("feature2_name", type=str, help="The name of the second feature.")
    parser.add_argument("--num_rows", type=int, default=10000, help="The number of rows in the CSV file. Default is 10000.")
    parser.add_argument("--correlation", type=float, default=0.9, help="The correlation coefficient between the features. Default is 0.9.")

    args = parser.parse_args()

    generate_correlated_csv(args.file_name, args.feature1_name, args.feature2_name, args.num_rows, args.correlation)

# Usage:
# python generate_csv.py correlated_data.csv Feature1 Feature2 --num_rows 5000 --correlation 0.95
