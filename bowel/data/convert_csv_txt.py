import argparse
import csv


def convert_csv_to_txt(csv_file, txt_file):
    """Converts csv file generated from model to txt file.

    Args:
        csv_file (src): Path to load csv file.
        txt_file (src): Path to save txt file.
    """
    with open(csv_file, 'r') as input, open(txt_file, 'w', newline='') as output:
        reader = csv.reader(input, delimiter=',')
        for row in reader:
            try:
                float(row[0])
            except ValueError:
                continue
            output.write(f'{row[0]}\t{row[1]}\t\n')
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='Path to csv annotation file')
    args = parser.parse_args()
    output_file = args.input_file.replace('.csv', '.txt')
    convert_csv_to_txt(args.input_file, output_file)
