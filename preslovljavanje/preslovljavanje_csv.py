import csv

# Function to convert text to "šišana latinica"
def convert_to_sisana_latinica(text):
    cyrillic_to_latin = {
        'č': 'c', 'ć': 'c', 'ž': 'z', 'š': 's', 'đ': 'dj',
        'Č': 'C', 'Ć': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'Dj'
    }
    return ''.join(cyrillic_to_latin.get(char, char) for char in text)

# Function to read, convert and save CSV file
def convert_csv_file(input_path, output_path, encoding='utf-8'):
    with open(input_path, mode='r', encoding=encoding, errors='replace') as infile, open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            converted_row = [convert_to_sisana_latinica(cell) for cell in row]
            writer.writerow(converted_row)

# Specify input and output file paths
input_csv_path = './data_csv/pripreme_za_cas_updated.csv'
output_csv_path = './data_csv/output3.csv'

# Convert and save the CSV file
convert_csv_file(input_csv_path, output_csv_path)
