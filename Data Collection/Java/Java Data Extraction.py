import os
from bs4 import BeautifulSoup
import csv
import re

top_directory = "api/"

total_entries = 0

extracted_data = []

meta_pattern = re.compile(r'declaration: module: ([^,]+), package: ([^,]+), class: ([^\s,]+)')

unique_descriptions = set()

for root, dirs, files in os.walk(top_directory):
    for filename in files:
        if filename.endswith(".html"):

            file_path = os.path.join(root, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                soup = BeautifulSoup(html_content, 'html.parser')

                meta_description = soup.find('meta', {'name': 'description'})

                if not meta_description:
                    continue

                meta_match = meta_pattern.search(meta_description['content'])
                if meta_match:
                    package_name = meta_match.group(2)
                    class_name = meta_match.group(3)
                else:
                    continue

                method_summary_section = soup.find('section', class_='method-summary')

                if method_summary_section:

                    summary_div = method_summary_section.find('div', class_='summary-table three-column-summary')

                    if summary_div:

                        member_links = summary_div.find_all('a', class_='member-name-link')

                        blocks = summary_div.find_all(class_='block')

                        if len(member_links) == len(blocks):
                            for i in range(len(member_links)):
                                method_name = member_links[i].get_text(strip=True)
                                description = blocks[i].get_text(strip=True).replace('\n', ' ').strip()  # Remove line breaks and extra whitespace

                                # Check if the description is unique
                                if description not in unique_descriptions:
                                    extracted_data.append({
                                        "Method Name": f"{package_name}.{class_name}.{method_name}",
                                        "Description": description
                                    })
                                    total_entries += 1
                                    unique_descriptions.add(description)

            except FileNotFoundError:
                print(f"The file '{file_path}' does not exist.")

            except IOError as e:
                print(f"An error occurred while reading the file: {e}")

print(f"Total Entries Extracted: {total_entries}")

csv_file_path = "extracted_data.csv"

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["Method Name", "Description"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for entry in extracted_data:
        writer.writerow(entry)

print(f"Data saved to {csv_file_path}")