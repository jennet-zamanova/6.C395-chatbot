from retrieval import find_matching_facilities, find_facility_by_name

queries = [
    "I need outpatient substance use treatment and I have MassHealth",
    "I'm looking for telehealth mental health care",
    "I need a Spanish-speaking program",
    "I'm a veteran looking for mental health support",
    "I need residential treatment for substance use",
]

for q in queries:
    print("\nQUERY:", q)
    results = find_matching_facilities(q, top_k=3)
    for r in results:
        print("-", r["display_name"])
        print("  reasons:", r["match_reasons"])
        print("  address:", r["address"])
        print("  phone:", r["phone"])

print(find_facility_by_name("Boston Medical Center", top_k=2))
print(find_facility_by_name("Bay Cove", top_k=2))