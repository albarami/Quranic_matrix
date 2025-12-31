# QBM v1.0.0 Academic Showcase

This directory contains exemplar outputs demonstrating the QBM system's capabilities for academic review.

## Files

| Question | JSON | Arabic | English | Capability |
|----------|------|--------|---------|------------|
| A01 | [A01.json](A01.json) | [A01.ar.md](A01.ar.md) | [A01.en.md](A01.en.md) | Causal Chain Analysis |
| A03 | [A03.json](A03.json) | [A03.ar.md](A03.ar.md) | [A03.en.md](A03.en.md) | Cycle Detection |
| A11 | [A11.json](A11.json) | [A11.ar.md](A11.ar.md) | [A11.en.md](A11.en.md) | Causal Strength |
| B02 | [B02.json](B02.json) | [B02.ar.md](B02.ar.md) | [B02.en.md](B02.en.md) | Tafsir Divergence |
| B11 | [B11.json](B11.json) | [B11.ar.md](B11.ar.md) | [B11.en.md](B11.en.md) | Disputed Behaviors |

## What Each Demonstrates

### A01: The Complete Destruction Pathway
- **Capability**: `GRAPH_CAUSAL`, `MULTIHOP`
- **Demonstrates**: Multi-hop causal chain tracing from heedlessness (الغفلة) to disbelief (الكفر)
- **Evidence**: Verse keys + tafsir citations for each link

### A03: Circular Reinforcement Detection
- **Capability**: `GRAPH_CAUSAL`, `MULTIHOP`
- **Demonstrates**: Finding self-reinforcing behavioral cycles (A→B→C→A)
- **Evidence**: Cycle paths with edge provenance

### A11: Causal Strength Quantification
- **Capability**: `GRAPH_CAUSAL`, `TAFSIR_MULTI_SOURCE`
- **Demonstrates**: Ranking causal claims by strength (verses × sources)
- **Evidence**: Quantified strength scores with source counts

### B02: The Riba Divergence
- **Capability**: `TAFSIR_MULTI_SOURCE`, `CONSENSUS`
- **Demonstrates**: Cross-source comparison of الربا interpretations
- **Evidence**: Divergence matrix across 5 tafsir sources

### B11: Disputed Behaviors
- **Capability**: `TAFSIR_MULTI_SOURCE`, `CONSENSUS`
- **Demonstrates**: Identifying contradictory interpretations
- **Evidence**: Dispute classification with source attribution

## Reproduction

To regenerate these outputs:

```bash
# From repository root
python scripts/generate_showcase.py
```

## Commit

These outputs were generated at commit `63785bf` (tag: v1.0.0).

---

*Generated: 2025-12-31*
