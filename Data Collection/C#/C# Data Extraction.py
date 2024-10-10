import os
import csv
from bs4 import BeautifulSoup

def extract_data_from_xml(xml_content):
    soup = BeautifulSoup(xml_content, 'xml')
    type_elements = soup.find_all('Type')

    extracted_data = []

    for type_element in type_elements:
        full_name = type_element.get('FullName')
        member_elements = type_element.find_all('Member')

        for member_element in member_elements:
            method_name = member_element.get('MemberName')

            if full_name and method_name:
                extracted_data.append(f"{full_name}.{method_name}")

    return extracted_data

def process_xml_files(xml_folder_path, csv_file_path):
    extracted_data = []

    for root, dirs, files in os.walk(xml_folder_path):
        for filename in files:
            if filename.endswith(".xml"):
                file_path = os.path.join(root, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        xml_content = file.read()

                    data_from_xml = extract_data_from_xml(xml_content)
                    extracted_data.extend(data_from_xml)

                except FileNotFoundError:
                    print(f"The file '{file_path}' does not exist.")

                except IOError as e:
                    print(f"An error occurred while reading the file: {e}")

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["FullName.MethodName"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the CSV header
        writer.writeheader()

        for entry in extracted_data:
            writer.writerow({"FullName.MethodName": entry})

    print(f"Data saved to {csv_file_path}")

# Example Usage
xml_folder_path = r"F:\Projects\Dada Vaia Projects\CrossAPILearning-master\data\APIDoc\dotnet-api-docs-main\dotnet-api-docs-main\xml"
csv_file_path = "extracted_data_modified.csv"

process_xml_files(xml_folder_path, csv_file_path)
