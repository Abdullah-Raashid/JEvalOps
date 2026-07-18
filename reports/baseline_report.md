# JEvalOps Evaluation Report

Model: `rule-based-japanese-enterprise-baseline`
Backend: `local_rule_based`
Composite quality score: **0.977**

## Task Scores
- `business_rewriting`: 1.000
- `information_extraction`: 1.000
- `summarization`: 1.000
- `grounded_qa`: 0.933
- `robustness`: 0.949

## Efficiency
- Mean latency seconds: 0.0010
- P95 latency seconds: 0.0010
- Mean tokens/sec: 36632.00

## Error Summary

## Japanese Error Taxonomy
- `language_errors`: incorrect_particle, unnatural_word_order, incorrect_kanji_selection, repetition, literal_translation_from_english
- `business_japanese_errors`: incorrect_honorific, excessive_honorific, inappropriate_politeness_level, wrong_internal_external_perspective, unnatural_corporate_phrasing
- `reasoning_errors`: missed_implicit_subject, date_normalization_error, numerical_interpretation_error, contradiction, instruction_omission
- `reliability_errors`: hallucination, unsupported_assumption, failure_to_abstain, invalid_json, privacy_sensitive_output
