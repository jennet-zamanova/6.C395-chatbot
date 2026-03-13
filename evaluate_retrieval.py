# Testing

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
        "required_attributes": ["opioid_treatment_program"],
        "criteria": {
            "location_city": "Boston",
            "substances": ["opioids"]
        }
    },
    {
        "query": "Mental health services for veterans",
        "required_attributes": ["veterans"],
        "criteria": {
            "special_populations": ["veterans"],
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Spanish speaking facilities in Boston",
        "required_attributes": ["spanish_support"],
        "criteria": {
            "languages": ["Spanish"],
            "location_city": "Boston"
        }
    },

    {
        "query": "I need telehealth mental health support",
        "required_attributes": ["telehealth", "mental_health"],
        "criteria": {
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Looking for residential substance use treatment",
        "required_attributes": ["residential", "substance_use"],
        "criteria": {
            "substances": ["alcohol", "drugs"]
        }
    },
    {
        "query": "I need inpatient care for substance use",
        "required_attributes": ["inpatient", "substance_use"],
        "criteria": {
            "treatment_type": "inpatient"
        }
    },
    {
        "query": "Need intensive outpatient addiction treatment",
        "required_attributes": ["intensive_outpatient", "substance_use"],
        "criteria": {
            "treatment_type": "intensive_outpatient"
        }
    },
    {
        "query": "Looking for partial hospitalization mental health services",
        "required_attributes": ["partial_hospitalization", "mental_health"],
        "criteria": {
            "treatment_type": "partial_hospitalization"
        }
    },
    {
        "query": "Need medication assisted treatment for opioid use",
        "required_attributes": ["medication_assisted_treatment", "substance_use"],
        "criteria": {
            "substances": ["opioids"]
        }
    },
    {
        "query": "Looking for CBT therapy in Boston",
        "required_attributes": ["cbt", "mental_health"],
        "criteria": {
            "location_city": "Boston"
        }
    },
    {
        "query": "I need DBT treatment near Boston",
        "required_attributes": ["dbt", "mental_health"],
        "criteria": {
            "location_city": "Boston"
        }
    },
    {
        "query": "Need care for adolescents with substance use issues",
        "required_attributes": ["adolescents", "substance_use"],
        "criteria": {
            "special_populations": ["adolescents"]
        }
    },
    {
        "query": "Looking for services for pregnant or postpartum patients",
        "required_attributes": ["pregnant_postpartum"],
        "criteria": {
            "special_populations": ["pregnant_postpartum"]
        }
    },
    {
        "query": "I need ASL-supported mental health treatment",
        "required_attributes": ["asl_support", "mental_health"],
        "criteria": {}
    },
    {
        "query": "Looking for Spanish-speaking addiction treatment with telehealth",
        "required_attributes": ["spanish_support", "telehealth", "substance_use"],
        "criteria": {
            "languages": ["Spanish"]
        }
    },
    {
        "query": "Need outpatient treatment that accepts Medicaid",
        "required_attributes": ["outpatient", "medicaid"],
        "criteria": {
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Looking for mental health treatment that accepts Medicare",
        "required_attributes": ["mental_health", "medicare"],
        "criteria": {}
    },
    {
        "query": "Need treatment covered by private insurance",
        "required_attributes": ["private_insurance"],
        "criteria": {}
    },
    {
        "query": "Looking for a facility that accepts TRICARE",
        "required_attributes": ["tricare"],
        "criteria": {}
    },
    {
        "query": "I need affordable treatment with sliding fee scale",
        "required_attributes": ["sliding_fee_scale"],
        "criteria": {}
    },
    {
        "query": "Looking for treatment that allows cash payment",
        "required_attributes": ["cash_payment"],
        "criteria": {}
    },
    {
        "query": "Need Boston outpatient mental health services in Spanish",
        "required_attributes": ["outpatient", "mental_health", "spanish_support"],
        "criteria": {
            "location_city": "Boston",
            "languages": ["Spanish"],
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Looking for Boston substance use treatment for veterans",
        "required_attributes": ["substance_use", "veterans"],
        "criteria": {
            "location_city": "Boston",
            "special_populations": ["veterans"]
        }
    },
    {
        "query": "Need residential care for pregnant patients with substance use issues",
        "required_attributes": ["residential", "pregnant_postpartum", "substance_use"],
        "criteria": {
            "special_populations": ["pregnant_postpartum"]
        }
    },
    {
        "query": "Need adolescent outpatient mental health treatment",
        "required_attributes": ["adolescents", "outpatient", "mental_health"],
        "criteria": {
            "special_populations": ["adolescents"],
            "treatment_type": "outpatient"
        }
    },
    {
        "query": "Looking for telehealth services that accept Medicaid",
        "required_attributes": ["telehealth", "medicaid"],
        "criteria": {}
    },
    {
        "query": "Need Boston inpatient mental health support",
        "required_attributes": ["inpatient", "mental_health"],
        "criteria": {
            "location_city": "Boston",
            "treatment_type": "inpatient"
        }
    },
    {
        "query": "Looking for Boston facilities with medication assisted treatment",
        "required_attributes": ["medication_assisted_treatment", "substance_use"],
        "criteria": {
            "location_city": "Boston",
            "substances": ["opioids"]
        }
    }
]

def evaluate_retrieval(query, required_attributes, criteria_dict=None, top_k=5):
    """
    Evaluate a single retrieval case based on required attributes

    Args:
        query (str): User query
        required_attributes (list): List of attributes that returned facilities must have
        criteria_dict (dict): Search criteria
        top_k (int): Return top k results

    Returns:
        dict: Evaluation metrics
    """

    # Convert criteria_dict to SearchCriteria object
    criteria = None
    if criteria_dict:
        criteria = SearchCriteria(**criteria_dict)


    try:
        results = find_matching_facilities(query, [], top_k=top_k, criteria=criteria)
        retrieved_facilities = [r.get('facility_name', '') for r in results]
    except Exception as e:
        print(f"Retrieval error: {e}")
        retrieved_facilities = []


    # Evaluate based on required attributes
    matching_count = 0
    for facility_name in retrieved_facilities:
        # Find facility in data
        matches = df[df['facility_name'].str.contains(facility_name, case=False, na=False)]
        if not matches.empty:
            # Check if facility has all required attributes
            has_all_attributes = True
            for attr in required_attributes:
                if attr in matches.columns and not matches[attr].any():
                    has_all_attributes = False
                    break
            if has_all_attributes:
                matching_count += 1

    # Calculate metrics
    precision = matching_count / len(retrieved_facilities) if retrieved_facilities else 0
    # For recall, we assume there are facilities with required attributes in the data
    facilities_with_attributes = 0
    for attr in required_attributes:
        if attr in df.columns:
            facilities_with_attributes = max(facilities_with_attributes, df[attr].sum())

    recall = matching_count / facilities_with_attributes if facilities_with_attributes > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "query": query,
        "retrieved": retrieved_facilities,
        "required_attributes": required_attributes,
        "matching_count": matching_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_with_attributes": facilities_with_attributes
    }

def run_evaluation():
 
    results = []

    print("Starting facility retrieval performance evaluation...")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {case['query']}")
        result = evaluate_retrieval(
            case['query'],
            case['required_attributes'],
            case.get('criteria'),
            top_k=5
        )
        results.append(result)

        print(f"Retrieved results: {result['retrieved']}")
        print(f"Required attributes: {result['required_attributes']}")
        print(f"Matching facilities: {result['matching_count']}/{len(result['retrieved'])}")
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


    # Save results to JSON (convert numpy types to native Python types)
    results_serializable = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy type
                serializable_result[key] = value.item()
            elif isinstance(value, list):
                serializable_result[key] = [v.item() if hasattr(v, 'item') else v for v in value]
            else:
                serializable_result[key] = value
        results_serializable.append(serializable_result)

    with open('evaluation_results_test.json', 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print("\nDetailed results saved to evaluation_results.json")

    return results

if __name__ == "__main__":
    run_evaluation()