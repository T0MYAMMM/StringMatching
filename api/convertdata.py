import pandas as pd

def convert_csv_to_json(csv_file_path, json_file_path):
    data = pd.read_csv(csv_file_path)
    data.to_json(json_file_path, orient='records')

csv_path = "rumahsakit_medicalclinicid_cleaned.csv"

convert_csv_to_json(csv_path, "rumahsakit_medicalclinicid_cleaned.json")