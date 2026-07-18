ERROR_TAXONOMY = {
    "language_errors": [
        "incorrect_particle",
        "unnatural_word_order",
        "incorrect_kanji_selection",
        "repetition",
        "literal_translation_from_english",
    ],
    "business_japanese_errors": [
        "incorrect_honorific",
        "excessive_honorific",
        "inappropriate_politeness_level",
        "wrong_internal_external_perspective",
        "unnatural_corporate_phrasing",
    ],
    "reasoning_errors": [
        "missed_implicit_subject",
        "date_normalization_error",
        "numerical_interpretation_error",
        "contradiction",
        "instruction_omission",
    ],
    "reliability_errors": [
        "hallucination",
        "unsupported_assumption",
        "failure_to_abstain",
        "invalid_json",
        "privacy_sensitive_output",
    ],
}

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
