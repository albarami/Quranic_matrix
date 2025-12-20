## Tafsir ingestion

### Minimum baseline set (v1)

- Ibn Kathir
- al-Tabari
- al-Sa‘di
- al-Qurtubi

### Target record shape (project contract)

```json
{
  "tafsir_id": "ibn_kathir",
  "reference": {"surah": 24, "ayah": 24},
  "text_ar": "...",
  "text_en": "..."
}
```

### Recommended local file naming

- `data/tafsir/ibn_kathir.ar.jsonl`
- `data/tafsir/saadi.ar.jsonl`

Use JSONL when you want line-delimited per-ayah records.


