# QBM Project - Task Checklist

## ✅ COMPLETED
- [x] Download Quran text (Tanzil XML)
- [x] Convert to tokenized JSON
- [x] Create vocabularies
- [x] Create coding manual
- [x] Create gold examples
- [x] Create Label Studio config

---

## 📍 CURRENT: PHASE 0 - PROJECT SETUP (Week 1)

### Day 1: GitHub Setup
- [ ] Create GitHub account (if needed)
- [ ] Create repository: `quranic-behavior-matrix`
- [ ] Clone locally:
  ```bash
  git clone https://github.com/YOUR_USERNAME/quranic-behavior-matrix.git
  cd quranic-behavior-matrix
  ```
- [ ] Create folder structure:
  ```bash
  mkdir -p {data/{raw,processed,annotations,exports},docs,src/{scripts,validation,api},config,tests,models,reports}
  ```
- [ ] Create .gitignore
- [ ] Initial commit:
  ```bash
  git add .
  git commit -m "chore: initial project structure"
  git push origin main
  ```

### Day 2: Add Documents
- [ ] Copy all docs to `docs/` folder
- [ ] Add README.md
- [ ] Commit:
  ```bash
  git add docs/ README.md
  git commit -m "docs: add specification and coding manual"
  git push origin main
  ```

### Day 3: Add Config Files
- [ ] Copy vocabularies to `config/`
- [ ] Copy Label Studio config to `config/`
- [ ] Copy gold examples to `config/`
- [ ] Set up Git LFS for large files:
  ```bash
  git lfs install
  git lfs track "*.json"
  ```
- [ ] Add data files to `data/`
- [ ] Commit:
  ```bash
  git add config/ data/ .gitattributes
  git commit -m "config: add vocabularies and data files"
  git push origin main
  ```

### Day 4: Python Environment
- [ ] Create virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- [ ] Install dependencies:
  ```bash
  pip install pandas numpy jsonschema scikit-learn fastapi uvicorn pytest
  pip freeze > requirements.txt
  ```
- [ ] Commit:
  ```bash
  git add requirements.txt
  git commit -m "chore: add Python dependencies"
  git push origin main
  ```

### Day 5: Label Studio
- [ ] Install: `pip install label-studio`
- [ ] Start: `label-studio start`
- [ ] Create project "QBM Pilot"
- [ ] Add labeling interface (copy XML from config)
- [ ] Import pilot data (50 ayat)
- [ ] Test annotation on 1-2 spans

### Day 6-7: Validation Scripts
- [ ] Create `src/validation/validate_schema.py`
- [ ] Create `src/validation/validate_vocabularies.py`
- [ ] Test scripts on gold examples
- [ ] Commit:
  ```bash
  git add src/
  git commit -m "feat: add validation scripts"
  git push origin main
  ```

### Phase 0 Complete ✓
```bash
git commit -m "milestone: complete Phase 0 - project setup"
git tag -a v0.1.0 -m "Phase 0 Complete"
git push origin main --tags
```

---

## 🔜 NEXT: PHASE 1 - SELF-CALIBRATION (Week 2)

### Day 1-2: Study
- [ ] Read coding manual cover to cover
- [ ] Study gold examples 1-10
- [ ] Note unclear sections

### Day 3-4: Blind Annotation
- [ ] Annotate 10 spans WITHOUT looking at answers
- [ ] Export your annotations

### Day 5-6: Compare & Learn
- [ ] Compare to gold examples
- [ ] Document mistakes
- [ ] Calculate personal accuracy

### Day 7: Update Manual
- [ ] Add clarifications based on difficulties
- [ ] Commit updates

### Phase 1 Complete ✓
```bash
git tag -a v0.2.0 -m "Phase 1 Complete: Self-Calibration"
git push origin main --tags
```

---

## 📅 FUTURE PHASES

### Phase 2: Micro-Pilot (Weeks 3-4)
- [ ] Recruit 2 annotators
- [ ] Train annotators (8 hours each)
- [ ] Triple-annotate 100 spans
- [ ] Calculate IAA
- [ ] Adjudicate disagreements

### Phase 3: Full Pilot (Weeks 5-8)
- [ ] Expand to 500 spans
- [ ] Distribute work
- [ ] Weekly check-ins
- [ ] Export Gold v0.1

### Phase 4: Scale-Up (Weeks 9-16)
- [ ] Expand team to 4-6 annotators
- [ ] Set up database
- [ ] Annotate 3,000+ spans
- [ ] Senior review process

### Phase 5: Production (Weeks 17-20)
- [ ] Build API
- [ ] Write documentation
- [ ] Run test suite
- [ ] Launch v1.0.0

### Phase 6: Maintenance (Ongoing)
- [ ] Monthly quality checks
- [ ] Quarterly expansions
- [ ] Academic publications

---

## 📊 KEY METRICS TO TRACK

| Metric | Target |
|--------|--------|
| IAA (Cohen's κ) | ≥ 0.70 |
| Gold spans | 2,000+ |
| Silver spans | 5,000+ |
| Coverage (surahs) | 80+ |

---

## 🔗 QUICK LINKS

- GitHub: https://github.com/YOUR_USERNAME/quranic-behavior-matrix
- Label Studio: http://localhost:8080
- Tanzil: https://tanzil.net

---

## 📞 CONTACTS

- Project Lead: Salim Al-Barami
- Foundational Research: Dr. Ibrahim Bouzidani

---

*Updated: December 2025*
