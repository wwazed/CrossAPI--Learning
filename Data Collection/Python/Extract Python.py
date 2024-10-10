import os
from bs4 import BeautifulSoup
import csv

def extract_dt_ids_from_html(file_path):
    # Load the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all <dl> tags with the specified classes
    dl_tags = soup.find_all('dl', class_=['py class', 'py method', 'py function'])
    
    # Extract the 'dt id' names
    dt_ids = []
    for dl in dl_tags:
        dt = dl.find('dt')
        if dt and 'id' in dt.attrs:
            dt_ids.append(dt['id'])
    
    return dt_ids

def process_html_files_in_folder(folder_path, output_csv_path):
    # Prepare to write to the CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dt id'])
        
        # Iterate through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.html'):
                file_path = os.path.join(folder_path, filename)
                # Extract dt ids from the HTML file
                dt_ids = extract_dt_ids_from_html(file_path)
                
                # Write dt ids to the CSV file
                for dt_id in dt_ids:
                    writer.writerow([dt_id])

# Define the folder path and output CSV file path
folder_path = 'F:\Projects\Dada Vaia Projects\CrossAPILearning-master\data\APIDoc\python-3.10.2-docs-html\python-3.10.2-docs-html\library'  # Replace with your folder path
output_csv_path = 'F:\Projects\Dada Vaia Projects\CrossAPILearning-master\data\APIDoc\python-3.10.2-docs-html\python-3.10.2-docs-html\library\Python.csv'

# Process the HTML files in the folder
process_html_files_in_folder(folder_path, output_csv_path)

output_csv_path
