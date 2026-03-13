"""
评估聊天机器人检索性能

此脚本评估设施检索功能的准确性，包括精确度、召回率和F1分数。
"""

import pandas as pd
from src.retrieval import find_matching_facilities
from src.search_criteria import SearchCriteria
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# 加载数据以获取实际设施名称
DATA_PATH = "data/boston_samhsa_clean.csv"
df = pd.read_csv(DATA_PATH)

# 定义测试案例：输入查询和预期设施（基于实际数据）
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
    评估单个检索案例

    Args:
        query (str): 用户查询
        expected_facilities (list): 预期设施名称列表
        criteria_dict (dict): 搜索标准
        top_k (int): 返回前k个结果

    Returns:
        dict: 评估指标
    """
    # 转换标准为SearchCriteria对象
    criteria = SearchCriteria()
    if criteria_dict:
        for key, value in criteria_dict.items():
            if hasattr(criteria, key):
                setattr(criteria, key, value)

    # 执行检索
    try:
        results = find_matching_facilities(query, [], top_k=top_k, criteria=criteria)
        retrieved_facilities = [r.get('facility_name', '') for r in results]
    except Exception as e:
        print(f"检索错误: {e}")
        retrieved_facilities = []

    # 计算指标
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
    """运行所有测试案例并计算平均指标"""
    results = []

    print("开始评估设施检索性能...")
    print("=" * 60)

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {case['query']}")
        result = evaluate_retrieval(
            case['query'],
            case['expected_facilities'],
            case.get('criteria'),
            top_k=5
        )
        results.append(result)

        print(f"检索结果: {result['retrieved']}")
        print(f"预期结果: {result['expected']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print("-" * 40)

    # 计算平均指标
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1_score'] for r in results) / len(results)

    print("\n总体评估结果:")
    print(".3f")
    print(".3f")
    print(".3f")

    # 保存结果到文件
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n详细结果已保存到 evaluation_results.json")

    return results

if __name__ == "__main__":
    run_evaluation()