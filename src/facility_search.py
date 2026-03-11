"""
Facility search module — stub for now.

Replace the search() implementation with the real SAMHSA/search logic.
Input: SearchCriteria from Chatbot.get_search_criteria()
Output: list of FacilityOption for Chatbot.explain_facility_options()
"""
from src.search_criteria import SearchCriteria, FacilityOption


def search(criteria: SearchCriteria) -> list[FacilityOption]:
    """
    Search for facilities matching the given criteria.
    TODO: Implement real search (e.g. SAMHSA API or local DB).
    For now returns mock options so the chatbot flow is testable.
    """
    # Stub: return mock options. Replace with real search.
    city = criteria.get("location_city") or "Boston"
    state = criteria.get("location_state") or "MA"
    treatment = criteria.get("treatment_type") or "outpatient"
    return [
        {
            "name": "Sample Counseling Center",
            "address": f"123 Main St, {city}, {state}",
            "phone": "(555) 123-4567",
            "treatment_types": [treatment, "telehealth"],
            "payment_options": ["Medicaid", "sliding scale", "private insurance"],
            "special_populations": ["adults"],
            "therapies": ["CBT", "12-step"],
            "languages": ["English", "Spanish"],
        },
        {
            "name": "Hope Recovery Center",
            "address": f"456 Oak Ave, {city}, {state}",
            "phone": "(555) 987-6543",
            "treatment_types": [treatment, "inpatient"],
            "payment_options": ["Medicaid", "free", "sliding scale"],
            "special_populations": ["veterans", "LGBTQ+"],
            "therapies": ["CBT", "MAT", "12-step"],
            "languages": ["English"],
        },
    ]
