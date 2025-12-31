# QBM Release and Operations Runbook

> **Version**: 1.0  
> **Updated**: 2025-12-31  
> **Applies to**: v1.0.0+

---

## 1. Running Locally

### CPU Mode (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Build KB without GPU proof
python scripts/build_kb.py --full --device cpu

# Run tests
python -m pytest tests/ -v --ignore=tests/tools/
```

### GPU Mode (Production)

```bash
# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Build KB with GPU proof
python scripts/build_kb.py --full --gpu-proof --device cuda

# Verify GPU proof
python scripts/gpu_proof_instrumentation.py --validate artifacts/audit_pack/gpu_proof/gpu_computation_proof.json
```

---

## 2. Azure Orchestration

### Required Environment Variables

```bash
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional
QBM_STRICT_MODE=1              # Enable strict verifier
QBM_REQUIRE_AUDIT_PACK=1       # Fail if audit pack missing (CI mode)
```

### Key Storage

**Never commit API keys to the repository.**

For local development:
```bash
# Create .env file (gitignored)
cp .env.example .env
# Edit .env with your keys
```

For CI/CD:
- Store keys in GitHub Secrets
- Reference as `${{ secrets.AZURE_OPENAI_API_KEY }}`

For production:
- Use Azure Key Vault
- Use managed identities where possible

---

## 3. Strict Mode Behavior

### What Triggers Rejection

| Violation | Description |
|-----------|-------------|
| `missing_structured_claims` | Response has >100 chars but no `claims[]` |
| `claim_without_evidence` | Claim has empty `supporting_verse_keys` |
| `invalid_verse_key` | Verse key not in SSOT (e.g., 999:999) |
| `narrative_invalid_claim_ref` | Narrative references non-existent claim_id |
| `surah_intro_in_tafsir` | Tafsir entry is surah introduction, not verse-specific |
| `subset_contract_violation` | Tafsir verse_keys not subset of response verse_keys |

### Fail-Closed Behavior

In strict mode (`QBM_STRICT_MODE=1`):
- Any violation causes the response to be rejected
- Verifier returns `is_valid=false` with violation details
- API returns 422 with violation list

In non-strict mode:
- Violations are logged as warnings
- Response is still returned
- Caller must check `warnings[]` in response

---

## 4. CI Artifacts

### Artifact Locations

| Job | Artifact Name | Contents |
|-----|---------------|----------|
| `audit-pack` | `audit-pack-${{ github.sha }}` | `audit_pack.zip` (CPU) |
| `gpu-release-build` | `gpu-release-${{ github.sha }}` | `audit_pack_gpu.zip` + `kb.zip` |

### Retention

- Default retention: **90 days**
- For permanent storage: attach to GitHub Release

### Downloading Artifacts

```bash
# Via GitHub CLI
gh run download <run-id> -n audit-pack-<sha>

# Via GitHub UI
# Actions → Workflow Run → Artifacts section
```

---

## 5. Secret Rotation

### Azure OpenAI Keys

1. Generate new key in Azure Portal
2. Update GitHub Secret: `Settings → Secrets → AZURE_OPENAI_API_KEY`
3. Update local `.env` file
4. Verify: `python -c "from src.api.main import app; print('OK')"`
5. Revoke old key in Azure Portal

### Rotation Schedule

| Secret | Rotation Frequency |
|--------|-------------------|
| Azure OpenAI API Key | Every 90 days |
| GitHub Actions tokens | Automatic (GITHUB_TOKEN) |

---

## 6. Release Checklist

Before tagging a release:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Benchmark 200/200: `pytest tests/test_benchmark_legendary.py -v`
- [ ] Clean git tree: `git status --porcelain` returns empty
- [ ] Audit pack valid: `python scripts/generate_audit_pack.py --strict`

After tagging:

- [ ] CI `audit-pack` job succeeds
- [ ] Download and verify artifact fields
- [ ] Attach `audit_pack.zip` to GitHub Release
- [ ] (Optional) Run `gpu-release-build` on self-hosted GPU runner
- [ ] (Optional) Attach `audit_pack_gpu.zip` to GitHub Release

---

## 7. Troubleshooting

### Audit Pack Fails with "Dirty Tree"

```bash
# Check what's uncommitted
git status --porcelain

# Commit or stash changes
git add -A && git commit -m "fix: ..."
# OR
git stash
```

### GPU Proof Invalid

```bash
# Check proof details
cat artifacts/audit_pack/gpu_proof/gpu_computation_proof.json | jq '.valid, .gpus_utilized, .max_utilization_percent'

# Ensure batch encoding is used (batch_size > 1)
# Ensure CUDA device is actually used (not CPU fallback)
```

### CI Artifact Missing

- Check if job completed successfully
- Check artifact retention (may have expired)
- Re-run workflow if needed

---

## 8. Contact

For operational issues, contact the repository maintainers.
