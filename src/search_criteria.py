"""
Structured types for facility search and results.

The facility search module (to be implemented separately) should:
- Accept SearchCriteria from the chatbot (e.g. from Chatbot.get_search_criteria()).
- Return a list of FacilityOption. The chatbot will explain these via explain_facility_options().
"""
from typing import TypedDict


class SearchCriteria(TypedDict, total=False):
    """Key information collected from the user for facility search. All fields optional."""
    location_city: str
    location_state: str
    location_zip: str
    treatment_type: str  # e.g. "inpatient", "outpatient", "residential", "telehealth"
    substances: list[str]  # substances addressed
    payment_options: list[str]  # e.g. "Medicaid", "sliding scale", "free", "private insurance"
    special_populations: list[str]  # e.g. "veterans", "LGBTQ+", "adolescents", "pregnant women"
    therapies: list[str]  # e.g. "CBT", "MAT", "12-step"
    languages: list[str]


class FacilityOption(TypedDict, total=False):
    """One facility option returned by the search module. Important features for the user."""
    name: str
    address: str
    phone: str
    treatment_types: list[str]
    payment_options: list[str]
    special_populations: list[str]
    therapies: list[str]
    languages: list[str]
    other: dict  # any other important features


LOCATION_KEYS = ("location_city", "location_state", "location_zip")


def ready_to_search(criteria: SearchCriteria) -> bool:
    """
    Return True if criteria has enough to run a facility search.
    Requires at least one location (city, state, or zip) and treatment_type.
    """
    has_location = any(criteria.get(k) for k in LOCATION_KEYS)
    has_treatment = bool(criteria.get("treatment_type"))
    return bool(has_location and has_treatment)
