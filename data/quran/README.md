## Quran text ingestion (tokenized)

### Target format (project contract)

The ingested Qur’an text should follow the structure described in `docs/startup/startup_requirements.md`:

- `quran_text_version`: e.g. `uthmani_hafs_v1`
- `tokenization_id`: e.g. `tok_v1`
- Surah → Ayah → Tokens with:
  - `index` (0-based)
  - `text`
  - `start_char` / `end_char` in the ayah string (char offsets)

### Freeze rule

Once pilot begins, **do not change** tokenization boundaries without bumping `tokenization_id` and re-deriving all token anchors.

### Recommended local file naming

- `data/quran/uthmani_hafs_v1.tok_v1.json`


