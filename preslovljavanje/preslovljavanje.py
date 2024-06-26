import pandas as pd

# Function to convert text to "šišana latinica"
def convert_to_sisana_latinica(text):
    cyrillic_to_latin = {
        'č': 'c', 'ć': 'c', 'ž': 'z', 'š': 's', 'đ': 'dj', 'dž': 'dz',
        'Č': 'C', 'Ć': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'Dj', 'Dž': 'Dz'
    }
    return ''.join(cyrillic_to_latin.get(char, char) for char in text)

# Load the Excel file
input_excel_path = 'data_csv/pripreme_za_cas.xlsx'
df = pd.read_excel(input_excel_path)

# Convert the dataframe content
df_converted = df.applymap(lambda x: convert_to_sisana_latinica(x) if isinstance(x, str) else x)

# Save the converted dataframe to a new Excel file
output_excel_path = 'data_csv/output.xlsx'
df_converted.to_excel(output_excel_path, index=False)