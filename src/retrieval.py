import pandas as pd

from src.search_criteria import SearchCriteria

DATA_PATH = "data/boston_samhsa_clean.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    bool_cols = [
        "mental_health",
        "substance_use",
        "telehealth",
        "residential",
        "partial_hospitalization",
        "outpatient",
        "intensive_outpatient",
        "inpatient",
        "cash_payment",
        "medicaid",
        "medicare",
        "private_insurance",
        "tricare",
        "sliding_fee_scale",
        "cbt",
        "dbt",
        "opioid_treatment_program",
        "medication_assisted_treatment",
        "veterans",
        "adolescents",
        "pregnant_postpartum",
        "asl_support",
        "spanish_support",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    text_cols = [
        "facility_name",
        "program_name",
        "display_name",
        "full_address",
        "phone",
        "website",
        "city",
        "state",
        "type_facility",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def extract_constraints(user_message, history=None, criteria: SearchCriteria | None=None):
    constraints = {
        "mental_health": False,
        "substance_use": False,
        "outpatient": False,
        "residential": False,
        "telehealth": False,
        "intensive_outpatient": False,
        "inpatient": False,
        "partial_hospitalization": False,
        "medicaid": False,
        "medicare": False,
        "private_insurance": False,
        "tricare": False,
        "sliding_fee_scale": False,
        "spanish_support": False,
        "asl_support": False,
        "veterans": False,
        "adolescents": False,
        "pregnant_postpartum": False,
        "cbt": False,
        "dbt": False,
        "opioid_treatment_program": False,
        "medication_assisted_treatment": False,
    }

    # location_city: str
    # location_state: str
    # location_zip: str

    # A) Fill from criteria if provided
    if criteria:
        # treatment_type
        t = (criteria.get("treatment_type") or "").lower()
        if "outpatient" in t:
            constraints["outpatient"] = True
        if "inpatient" in t:
            constraints["inpatient"] = True
        if "residential" in t:
            constraints["residential"] = True
        if "telehealth" in t:
            constraints["telehealth"] = True
        # payment_options
        pays = [p.lower() for p in criteria.get("payment_options", [])]
        if any("medicaid" in p for p in pays):
            constraints["medicaid"] = True
        if any("sliding" in p for p in pays):
            constraints["sliding_fee_scale"] = True
        if any("private" in p or "insurance" in p for p in pays):
            constraints["private_insurance"] = True
        # special_populations
        pops = [p.lower() for p in criteria.get("special_populations", [])]
        if any("veteran" in p for p in pops):
            constraints["veterans"] = True
        if any("adolescent" in p or "teen" in p for p in pops):
            constraints["adolescents"] = True
        if any("pregnant" in p or "postpartum" in p for p in pops):
            constraints["pregnant_postpartum"] = True
        # therapies
        ths = [t.lower() for t in criteria.get("therapies", [])]
        if any("cbt" in t for t in ths):
            constraints["cbt"] = True
        if any("dbt" in t for t in ths):
            constraints["dbt"] = True
        if any("mat" in t or "medication" in t for t in ths):
            constraints["medication_assisted_treatment"] = True
        # languages
        langs = [l.lower() for l in criteria.get("languages", [])]
        if any("spanish" in l for l in langs):
            constraints["spanish_support"] = True
        if any("asl" in l or "sign" in l for l in langs):
            constraints["asl_support"] = True

        substances = [s.lower() for s in criteria.get("substances", [])]
        if any("opioid" in s for s in substances):
            constraints["opioid_treatment_program"] = True
        if len(substances) != 0:
            constraints["substance_use"] = True

    mh_keywords = [
        "mental health",
        "depression",
        "anxiety",
        "therapy",
        "therapist",
        "psychiatric",
        "psychiatry",
        "counseling",
        "counselling",
        "bipolar",
        "ptsd",
        "trauma",
    ]
    su_keywords = [
        "substance use",
        "addiction",
        "alcohol",
        "drug",
        "drugs",
        "opioid",
        "opioids",
        "recovery",
        "sobriety",
        "detox",
        "sober",
        "use disorder",
    ]

    texts = []
    if history:
        for message in history:
            if message["role"] == "user":
                texts.append(str(message["content"]))
    texts.append(user_message)
    text = " ".join(t.lower() for t in texts)

    if any(k in text for k in mh_keywords):
        constraints["mental_health"] = True
    if any(k in text for k in su_keywords):
        constraints["substance_use"] = True

    if "outpatient" in text:
        constraints["outpatient"] = True
    if "residential" in text or "live there" in text or "recovery home" in text:
        constraints["residential"] = True
    if "telehealth" in text or "virtual" in text or "online" in text or "remote" in text:
        constraints["telehealth"] = True
    if "intensive outpatient" in text or "iop" in text:
        constraints["intensive_outpatient"] = True
    if "inpatient" in text or "hospital" in text:
        constraints["inpatient"] = True
    if "partial hospitalization" in text or "php" in text:
        constraints["partial_hospitalization"] = True

    if "masshealth" in text or "medicaid" in text:
        constraints["medicaid"] = True
    if "medicare" in text:
        constraints["medicare"] = True
    if "private insurance" in text or "commercial insurance" in text:
        constraints["private_insurance"] = True
    if "tricare" in text:
        constraints["tricare"] = True
    if "sliding scale" in text or "sliding fee" in text or "low cost" in text or "affordable" in text:
        constraints["sliding_fee_scale"] = True

    if "spanish" in text:
        constraints["spanish_support"] = True
    if "asl" in text or "sign language" in text:
        constraints["asl_support"] = True

    if "veteran" in text or "military" in text:
        constraints["veterans"] = True
    if "adolescent" in text or "teen" in text or "teens" in text or "youth" in text:
        constraints["adolescents"] = True
    if "pregnant" in text or "postpartum" in text:
        constraints["pregnant_postpartum"] = True

    if "cbt" in text:
        constraints["cbt"] = True
    if "dbt" in text:
        constraints["dbt"] = True
    if "otp" in text:
        constraints["opioid_treatment_program"] = True
    if "mat" in text or "medication assisted treatment" in text or "medication-assisted treatment" in text:
        constraints["medication_assisted_treatment"] = True

    return constraints


def filter_facilities(df, constraints):
    result = df.copy()

    if constraints["mental_health"] and not constraints["substance_use"]:
        result = result[result["mental_health"]]
    elif constraints["substance_use"] and not constraints["mental_health"]:
        result = result[result["substance_use"]]
    elif constraints["mental_health"] and constraints["substance_use"]:
        result = result[result["mental_health"] | result["substance_use"]]

    hard_cols = [
        "outpatient",
        "residential",
        "telehealth",
        "intensive_outpatient",
        "inpatient",
        "partial_hospitalization",
        "medicaid",
        "medicare",
        "private_insurance",
        "tricare",
    ]

    for col in hard_cols:
        if constraints[col]:
            result = result[result[col]]

    return result


# scoring function to rank facilities based on how many constraints they match, with some attributes weighted more heavily
# can be modified to incorporate more complex logic, e.g. partial matches, location preferences, etc.
def rank_facilities(df, constraints):
    ranked = df.copy()
    ranked["score"] = 0.0

    weights = {
        "mental_health": 3.0,
        "substance_use": 3.0,
        "outpatient": 3.0,
        "residential": 3.0,
        "telehealth": 3.0,
        "intensive_outpatient": 3.0,
        "inpatient": 3.0,
        "partial_hospitalization": 3.0,
        "medicaid": 2.0,
        "medicare": 2.0,
        "private_insurance": 2.0,
        "tricare": 2.0,
        "sliding_fee_scale": 1.5,
        "spanish_support": 1.5,
        "asl_support": 1.5,
        "veterans": 1.5,
        "adolescents": 1.5,
        "pregnant_postpartum": 1.5,
        "cbt": 1.0,
        "dbt": 1.0,
        "opioid_treatment_program": 1.0,
        "medication_assisted_treatment": 1.0,
    }

    for col, weight in weights.items():
        if constraints[col] and col in ranked.columns:
            ranked["score"] += ranked[col].astype(int) * weight

    # location preference: prioritize Boston proper and nearby areas
    city_weights = {
        "Boston": 3.0,
        "Charlestown": 2.5,
        "Dorchester": 2.5,
        "Dorchester Center": 2.5,
        "Jamaica Plain": 2.5,
        "Roxbury": 2.5,
        "Roslindale": 2.5,
        "Mattapan": 2.5,
        "Hyde Park": 2.5,
        "Brighton": 2.0,
        "Allston": 2.0,
        "East Boston": 2.0,
        "South Boston": 2.0,
        "Brookline": 1.5,
        "Cambridge": 1.5,
        "Somerville": 1.5,
        "Quincy": 1.0,
        "Waltham": 0.5,
        "Framingham": 0.0,
        "Marlborough": 0.0,
        "Wakefield": 0.0,
    }
    ranked["score"] += ranked["city"].map(city_weights).fillna(0.5)

    # query-aware boosts
    if constraints["substance_use"]:
        ranked["score"] += ranked["medication_assisted_treatment"].astype(int) * 1.0
        ranked["score"] += ranked["opioid_treatment_program"].astype(int) * 0.8

    if constraints["mental_health"]:
        ranked["score"] += ranked["cbt"].astype(int) * 0.6
        ranked["score"] += ranked["dbt"].astype(int) * 0.6

    # slight preference for rows with websites/phones filled in
    ranked["score"] += ranked["website"].astype(str).ne("").astype(int) * 0.2
    ranked["score"] += ranked["phone"].astype(str).ne("").astype(int) * 0.2

    ranked = ranked.sort_values(
        by=["score", "city", "facility_name", "program_name"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    return ranked


def build_match_reasons(row, constraints):
    reasons = []

    mapping = [
        ("mental_health", "offers mental health services"),
        ("substance_use", "offers substance use treatment"),
        ("outpatient", "offers outpatient care"),
        ("residential", "offers residential care"),
        ("telehealth", "offers telehealth"),
        ("intensive_outpatient", "offers intensive outpatient care"),
        ("inpatient", "offers inpatient care"),
        ("partial_hospitalization", "offers partial hospitalization"),
        ("medicaid", "accepts Medicaid / MassHealth"),
        ("medicare", "accepts Medicare"),
        ("private_insurance", "accepts private insurance"),
        ("tricare", "accepts TRICARE"),
        ("sliding_fee_scale", "offers sliding fee scale"),
        ("spanish_support", "offers Spanish support"),
        ("asl_support", "offers ASL support"),
        ("veterans", "serves veterans"),
        ("adolescents", "serves adolescents"),
        ("pregnant_postpartum", "serves pregnant or postpartum clients"),
        ("cbt", "offers CBT"),
        ("dbt", "offers DBT"),
        ("opioid_treatment_program", "offers an opioid treatment program"),
        ("medication_assisted_treatment", "offers medication-assisted treatment"),
    ]

    for col, message in mapping:
        if constraints.get(col, False) and bool(row.get(col, False)):
            reasons.append(message)

    if not reasons:
        fallback = [
            ("mental_health", "offers mental health services"),
            ("substance_use", "offers substance use treatment"),
            ("outpatient", "offers outpatient care"),
            ("telehealth", "offers telehealth"),
            ("residential", "offers residential care"),
        ]
        for col, message in fallback:
            if bool(row.get(col, False)):
                reasons.append(message)

    return reasons[:4]


def format_results(df, constraints, top_k=5):
    results = []

    attribute_cols = [
        "mental_health",
        "substance_use",
        "outpatient",
        "residential",
        "telehealth",
        "intensive_outpatient",
        "inpatient",
        "partial_hospitalization",
        "medicaid",
        "medicare",
        "private_insurance",
        "tricare",
        "sliding_fee_scale",
        "spanish_support",
        "asl_support",
        "veterans",
        "adolescents",
        "pregnant_postpartum",
        "cbt",
        "dbt",
        "opioid_treatment_program",
        "medication_assisted_treatment",
    ]

    for _, row in df.head(top_k).iterrows():
        attributes = {col: bool(row.get(col, False)) for col in attribute_cols}

        results.append(
            {
                "display_name": row.get("display_name", ""),
                "facility_name": row.get("facility_name", ""),
                "program_name": row.get("program_name", ""),
                "address": row.get("full_address", ""),
                "phone": row.get("phone", ""),
                "website": row.get("website", ""),
                "city": row.get("city", ""),
                "state": row.get("state", ""),
                "match_score": float(row.get("score", 0.0)),
                "match_reasons": build_match_reasons(row, constraints),
                "attributes": attributes,
            }
        )

    return results


def find_matching_facilities(user_message, history=None, top_k=5, criteria: SearchCriteria | None=None):
    df = load_data()
    constraints = extract_constraints(user_message, history, criteria)
    filtered = filter_facilities(df, constraints)

    if filtered.empty:
        ranked = rank_facilities(df, constraints)
        results = format_results(ranked, constraints, top_k=top_k)
        for result in results:
            result["note"] = "No exact match found; this is a close match."
        return results

    ranked = rank_facilities(filtered, constraints)
    return format_results(ranked, constraints, top_k=top_k)


def find_facility_by_name(name_query, top_k=3):
    df = load_data()
    q = name_query.lower().strip()

    if not q:
        return []

    candidates = df[
        df["display_name"].str.lower().str.contains(q, na=False)
        | df["facility_name"].str.lower().str.contains(q, na=False)
        | df["program_name"].str.lower().str.contains(q, na=False)
    ].copy()

    if candidates.empty:
        return []

    candidates["name_score"] = 0.0
    candidates["name_score"] += candidates["display_name"].str.lower().str.contains(q, na=False).astype(int) * 3.0
    candidates["name_score"] += candidates["facility_name"].str.lower().str.contains(q, na=False).astype(int) * 2.0
    candidates["name_score"] += candidates["program_name"].str.lower().str.contains(q, na=False).astype(int) * 1.0

    # slight preference for Boston proper here
    city_weights = {
        "Boston": 3.0,
        "Charlestown": 2.5,
        "Dorchester": 2.5,
        "Dorchester Center": 2.5,
        "Jamaica Plain": 2.5,
        "Roxbury": 2.5,
        "Roslindale": 2.5,
        "Mattapan": 2.5,
        "Hyde Park": 2.5,
        "Brighton": 2.0,
        "Allston": 2.0,
        "East Boston": 2.0,
        "South Boston": 2.0,
        "Brookline": 1.5,
        "Cambridge": 1.5,
        "Somerville": 1.5,
        "Quincy": 1.0,
        "Waltham": 0.5,
        "Framingham": 0.0,
        "Marlborough": 0.0,
        "Wakefield": 0.0,
    }
    candidates["name_score"] += candidates["city"].map(city_weights).fillna(0.5)

    candidates = candidates.sort_values(
        by=["name_score", "city", "facility_name", "program_name"],
        ascending=[False, True, True, True],
        na_position="last",
    )

    candidates["score"] = candidates["name_score"]

    # no specific constraints here since this is a direct lookup
    return format_results(candidates, constraints={}, top_k=top_k)