import numpy as np
import json

def safe_json(obj):
    # Scalars (NumPy → Python)
    if isinstance(obj, np.generic):
        return obj.item()

    # Arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Dicts
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}

    # Lists / Tuples / Sets
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(x) for x in obj]

    # Everything else (assumed already serializable)
    return obj

def safe_dump(obj, fn, indent=2):
    with open(fn, 'w') as f:
        json.dump(safe_json(obj), f,indent=indent)
def safe_print(obj, indent=2):
    print(json.dumps(safe_json(obj), indent=indent))

