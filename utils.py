import os
import json

def load_json_mapping(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: Mapping file not found at {filepath}")
    return {}
