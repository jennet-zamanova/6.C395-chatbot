

import pandas as pd
from src.retrieval import find_matching_facilities
from src.search_criteria import SearchCriteria
from sklearn.metrics import precision_score, recall_score, f1_score
import json


DATA_PATH = "data/boston_samhsa_clean.csv"
df = pd.read_csv(DATA_PATH)


test_cases = [
    {
        "query": "I need opioid treatment in Boston",
        "expected_facilities": ["Bay Cove Treatment Center", "Boston Medical Center"],  # 这些设施有opioid_treatment_program=True
        "criteria": {
            "location_city": "Boston",
            "substances": ["opioids"]
        }
    },
    {
        "query": "Mental health services for veterans",
        "expected_facilities": ["Bay Cove Treatment Center", "Boston Medical Center"],  # 这些有veterans=True
        "criteria": {
            "special_populations": ["veterans"],
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Spanish speaking facilities in Boston",
        "expected_facilities": ["Boston Medical Center", "Casa Esperanza Inc"],  # 这些有spanish_support=True
        "criteria": {
            "languages": ["Spanish"],
            "location_city": "Boston"
        }
    }
]

def evaluate_retrieval(query, expected_facilities, criteria_dict=None, top_k=5):
    """
    Evaluate a single retrieval case

    Args:
        query (str): User query
        expected_facilities (list): List of expected facility names
        criteria_dict (dict): Search criteria
        top_k (int): Return top k results

    Returns:
        dict: Evaluation metrics
    """

    criteria = SearchCriteria()
    if criteria_dict:
        for key, value in criteria_dict.items():
            if hasattr(criteria, key):
                setattr(criteria, key, value)


    try:
        results = find_matching_facilities(query, [], top_k=top_k, criteria=criteria)
        retrieved_facilities = [r.get('facility_name', '') for r in results]
    except Exception as e:
        print(f"Retrieval error: {e}")
        retrieved_facilities = []


    retrieved_set = set(retrieved_facilities)
    expected_set = set(expected_facilities)

    true_positives = len(retrieved_set & expected_set)
    false_positives = len(retrieved_set - expected_set)
    false_negatives = len(expected_set - retrieved_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "query": query,
        "retrieved": retrieved_facilities,
        "expected": expected_facilities,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def run_evaluation():
 
    results = []

    print("Starting facility retrieval performance evaluation...")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {case['query']}")
        result = evaluate_retrieval(
            case['query'],
            case['expected_facilities'],
            case.get('criteria'),
            top_k=5
        )
        results.append(result)

        print(f"Retrieved results: {result['retrieved']}")
        print(f"Expected results: {result['expected']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print("-" * 40)


    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1_score'] for r in results) / len(results)

    print("\nOverall evaluation results:")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1 Score: {avg_f1:.3f}")


    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDetailed results saved to evaluation_results.json")

    return results

if __name__ == "__main__":
    run_evaluation()