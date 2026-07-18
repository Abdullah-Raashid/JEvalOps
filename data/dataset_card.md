# JEvalOps Dataset Card

## Purpose

JEvalOps is a provenance-tracked Japanese enterprise benchmark for evaluating compact LLM behavior across business rewriting, information extraction, summarization, grounded QA, and robustness/normalization tasks.

## Governance

- License: project-owned template data by default.
- PII policy: records containing emails, phone numbers, postal codes, or Japanese My Number-like identifiers are rejected by validation.
- Test policy: generated test records are marked human-authored/project-authored and should be frozen before model adaptation.
- Synthetic policy: synthetic records are allowed in train/validation only and must pass quality filters.
- Review status: every accepted record carries `review_status`, `source_type`, `source_name`, `creation_method`, and `dataset_version`.

## Limitations

- Template-generated examples are useful for reproducibility but do not replace native-speaker annotation.
- Japanese politeness has multiple valid answers, so rubric review remains important.
- Enterprise domains need domain-expert review before production use.
- Automated metrics do not fully capture naturalness, cultural appropriateness, or nuanced keigo.
