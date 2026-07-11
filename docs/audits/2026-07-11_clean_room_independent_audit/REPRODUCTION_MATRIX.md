# Reproduction matrix

| Layer | Replay status | Evidence | Blocker |
|---|---|---|---|
| Static config parsing | Reproducible | all three TC1 configs load | none |
| Statistical claim calculation | Partly reproducible | audit transcript reported 62/62 report-claim checks | depends on committed derived artifacts |
| Unit/integration tests | Reproducible locally | audit transcript reported 339 pytest tests | CI does not execute them |
| Main NF4 execution | Partly reproducible | current config pins model revisions | historical raw/summary provenance incomplete |
| INT8 precision execution | Not fully reproducible | jobs/config present | model revisions absent |
| HarmBench primary re-score | Not fully reproducible | scorer implementation/job present | judge revision absent |
| API second judge | Not fully reproducible | sidecars and model name present | mutable service/version and data-transfer record absent |
| Exact historical rerun | UNVERIFIED | immutable tree and hashes exist | environment/dataset/cache provenance absent |
