import csv
import os

def search_csv(queries):
    csv_path = r"c:\Users\HP\Desktop\nutri\nutrition.csv"
    matches = []
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].lower()
            for q in queries:
                if q.lower() in name:
                    matches.append(row['name'])
                    if len(matches) >= 10:
                        return matches
    return matches

if __name__ == "__main__":
    print(search_csv(["soda", "cola", "pepsi", "drink"]))
