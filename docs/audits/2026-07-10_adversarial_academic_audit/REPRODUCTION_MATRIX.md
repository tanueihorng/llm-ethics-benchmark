# Reproduction matrix

| Layer | Evidence available | Supported claim | Gap/action |
|---|---|---|---|
| Results | Immutable 512 raw/summary tree + 300-file manifest | Existing evidence integrity. | Not an environment/data snapshot. |
| Judge scoring | Redacted judge sidecars + analysis JSON | Recompute current ASR aggregates/paired contrasts. | Document commands and digests. |
| Report numbers | `verify_report_claims.py` | 61 selected report/thesis/interim claims match artifacts. | Not method validity or humanized-document semantic proof. |
| Model identity | Post-run revision config fields | Future requested revision identified. | Raw/summary lack revision; pins committed after results. |
| Dataset identity | Dataset/config/split names | Intended inputs identified. | No Hub revision/fingerprint; unpinned loader calls. |
| Runtime | Point-in-time prose + lower-bound requirements | Broad dependency families known. | No lock/freeze or immutable CUDA/driver/library record. |
| Resume | Prompt-ID skip | Resume when unchanged. | No condition fingerprint; mixed-condition result possible. |
| TC1 execution | Canonical sbatch uses offline flags/config | Local-cache procedure documented. | Public `cluster-submit` violates policy/depends on ignored manifests. |
| CI | Local gates | Manual policy checks available. | CI skips pytest/claim lock. |

## Future-run manifest

Before accepting any raw record, save config bytes/SHA, code SHA, model/tokenizer revisions, dataset revision/fingerprint, quant method/dtype, seed/decode config, package lock, Python/CUDA/driver/GPU metadata, command/timestamps and file digests. Resume must require the same fingerprint.

Do not modify historical raw evidence. Publish a derived provenance note separating verified current facts from unverified historical identity.
