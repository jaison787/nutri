import csv
import os
import re

class NutritionLookup:
    def __init__(self, csv_path):
        self.data = []
        self.csv_path = csv_path
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV not found at {self.csv_path}")
            return

        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Basic cleaning and storage
                    self.data.append({
                        "name": row['name'],
                        "calories": self._to_int(row['calories']),
                        "protein": row['protein'],
                        "carbs": row['carbohydrate'],
                        "fat": row['total_fat']
                    })
            print(f"Loaded {len(self.data)} items from nutrition CSV.")
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def _to_int(self, val):
        try:
            # Remove any non-numeric chars except digits
            clean_val = re.sub(r'[^\d]', '', val)
            return int(clean_val) if clean_val else 0
        except:
            return 0

    def find_food(self, query):
        if not query:
            return None
        
        query = query.lower()
        matches = []
        
        # 1. Look for exact substring matches
        for item in self.data:
            item_name = item['name'].lower()
            if query == item_name:
                return item # Exact match is best
            
            if query in item_name:
                # Basic score: ratio of lengths
                score = len(query) / len(item_name)
                
                # Bonus if query is a standalone word
                if re.search(r'\b' + re.escape(query) + r'\b', item_name):
                    score += 1.0
                
                # Bonus if it starts with the query
                if item_name.startswith(query):
                    score += 0.5
                    
                matches.append((item, score))
        
        if matches:
            # Sort by score descending (higher ratio first)
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]
            
        return None

# Singleton instance
current_dir = os.path.dirname(os.path.abspath(__file__))
# The CSV is one level up from backend/
csv_path = os.path.join(current_dir, "..", "nutrition.csv")
lookup = NutritionLookup(csv_path)

def get_nutrition(food_name):
    return lookup.find_food(food_name)
