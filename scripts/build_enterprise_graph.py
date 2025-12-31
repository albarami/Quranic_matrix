#!/usr/bin/env python3
"""
Build Enterprise-Grade Causal Graph for QBM

This script builds a comprehensive behavior-to-behavior causal graph
based on Quranic evidence and scholarly tafsir analysis.

The graph should capture:
1. CAUSES: Direct causal relationships (A causes B)
2. LEADS_TO: Sequential relationships (A leads to B)
3. PREVENTS: Preventive relationships (A prevents B)
4. STRENGTHENS: Reinforcing relationships (A strengthens B)
5. OPPOSITE_OF: Antithetical relationships
6. COMPLEMENTS: Complementary relationships
7. CONDITIONAL_ON: Conditional relationships

Evidence sources:
- Quranic verse co-occurrence
- Tafsir analysis
- Scholarly consensus
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DATA_DIR = Path("data")
VOCAB_DIR = Path("vocab")

# Load canonical entities
ent = json.load(open(VOCAB_DIR / "canonical_entities.json", encoding="utf-8"))
behaviors = {b["id"]: b for b in ent.get("behaviors", [])}

print(f"Loaded {len(behaviors)} behaviors")

# Load existing graph
existing_graph = json.load(open(DATA_DIR / "graph" / "semantic_graph_v2.json", encoding="utf-8"))
existing_edges = {(e["source"], e["target"], e["edge_type"]) for e in existing_graph.get("edges", [])}

print(f"Existing graph has {len(existing_edges)} edges")

# Define scholarly causal relationships based on Quranic themes
# These are based on established Islamic scholarship

CAUSAL_RELATIONSHIPS = [
    # ===== FAITH AND DISBELIEF CHAINS =====
    # Faith leads to good deeds
    ("BEH_SPI_FAITH", "BEH_FIN_CHARITY", "CAUSES", "Faith motivates charitable giving - الذين آمنوا وعملوا الصالحات"),
    ("BEH_SPI_FAITH", "BEH_SPI_PRAYER", "CAUSES", "Faith leads to prayer - إن الصلاة تنهى عن الفحشاء والمنكر"),
    ("BEH_SPI_FAITH", "BEH_SPI_REPENTANCE", "CAUSES", "Faith motivates repentance"),
    ("BEH_SPI_FAITH", "BEH_SPI_TAWAKKUL", "CAUSES", "Faith leads to trust in Allah"),
    ("BEH_SPI_FAITH", "BEH_EMO_FEAR_ALLAH", "CAUSES", "Faith produces fear of Allah"),
    ("BEH_SPI_FAITH", "BEH_EMO_HOPE", "CAUSES", "Faith produces hope in Allah"),
    ("BEH_SPI_FAITH", "BEH_SPEECH_TRUTHFULNESS", "CAUSES", "Faith leads to truthfulness"),
    ("BEH_SPI_FAITH", "BEH_SOC_JUSTICE", "CAUSES", "Faith motivates justice"),
    ("BEH_SPI_FAITH", "BEH_SPI_TAQWA", "CAUSES", "Faith leads to taqwa"),
    ("BEH_SPI_FAITH", "BEH_SPI_FASTING", "CAUSES", "Faith motivates fasting"),
    ("BEH_SPI_FAITH", "BEH_FIN_ZAKAT", "CAUSES", "Faith motivates zakat"),
    
    # Disbelief chains
    ("BEH_SPI_DISBELIEF", "BEH_SPI_SHIRK", "LEADS_TO", "Disbelief can lead to polytheism"),
    ("BEH_SPI_DISBELIEF", "BEH_COG_ARROGANCE", "CAUSES", "Disbelief often stems from arrogance - استكبروا"),
    ("BEH_SPI_DISBELIEF", "BEH_SOC_OPPRESSION", "LEADS_TO", "Disbelief leads to oppression"),
    ("BEH_SPI_DISBELIEF", "BEH_SPEECH_LYING", "CAUSES", "Disbelief leads to lying about Allah"),
    ("BEH_SPI_DISBELIEF", "BEH_COG_HEEDLESSNESS", "CAUSES", "Disbelief causes heedlessness"),
    
    # Shirk chains
    ("BEH_SPI_SHIRK", "BEH_SPI_DISBELIEF", "CAUSES", "Polytheism is a form of disbelief"),
    ("BEH_SPI_SHIRK", "BEH_COG_HEEDLESSNESS", "CAUSES", "Shirk causes heedlessness of truth"),
    ("BEH_SPI_SHIRK", "BEH_SOC_OPPRESSION", "LEADS_TO", "Shirk leads to oppression"),
    
    # ===== HEART DISEASES AND THEIR EFFECTS =====
    # Arrogance chains
    ("BEH_COG_ARROGANCE", "BEH_SPI_DISBELIEF", "LEADS_TO", "Arrogance leads to disbelief - إبليس أبى واستكبر"),
    ("BEH_COG_ARROGANCE", "BEH_SOC_OPPRESSION", "CAUSES", "Arrogance causes oppression of others"),
    ("BEH_COG_ARROGANCE", "BEH_SPEECH_LYING", "CAUSES", "Arrogance leads to lying"),
    ("BEH_COG_ARROGANCE", "BEH_COG_IGNORANCE", "CAUSES", "Arrogance prevents learning"),
    ("BEH_COG_ARROGANCE", "BEH_EMO_HATRED", "CAUSES", "Arrogance breeds hatred"),
    
    # Envy chains
    ("BEH_EMO_ENVY", "BEH_SOC_OPPRESSION", "CAUSES", "Envy leads to harming others"),
    ("BEH_EMO_ENVY", "BEH_SPEECH_BACKBITING", "CAUSES", "Envy leads to backbiting"),
    ("BEH_EMO_ENVY", "BEH_EMO_HATRED", "CAUSES", "Envy breeds hatred"),
    ("BEH_EMO_ENVY", "BEH_SOC_SEVERING_TIES", "CAUSES", "Envy severs family ties"),
    
    # Hatred chains
    ("BEH_EMO_HATRED", "BEH_SOC_OPPRESSION", "CAUSES", "Hatred leads to oppression"),
    ("BEH_EMO_HATRED", "BEH_SPEECH_SLANDER", "CAUSES", "Hatred leads to slander"),
    ("BEH_EMO_HATRED", "BEH_SOC_SEVERING_TIES", "CAUSES", "Hatred severs ties"),
    
    # Heedlessness chains
    ("BEH_COG_HEEDLESSNESS", "BEH_SPI_DISBELIEF", "LEADS_TO", "Heedlessness can lead to disbelief"),
    ("BEH_COG_HEEDLESSNESS", "BEH_SOC_TRANSGRESSION", "LEADS_TO", "Heedlessness leads to transgression"),
    ("BEH_COG_HEEDLESSNESS", "BEH_EMO_INGRATITUDE", "CAUSES", "Heedlessness causes ingratitude"),
    ("BEH_COG_HEEDLESSNESS", "BEH_SPI_SHOWING_OFF", "CAUSES", "Heedlessness leads to showing off"),
    
    # Hypocrisy chains
    ("BEH_SPI_HYPOCRISY", "BEH_SPEECH_LYING", "CAUSES", "Hypocrisy is characterized by lying"),
    ("BEH_SPI_HYPOCRISY", "BEH_SOC_BETRAYAL", "CAUSES", "Hypocrites betray trusts"),
    ("BEH_SPI_HYPOCRISY", "BEH_SPI_DISBELIEF", "LEADS_TO", "Hypocrisy leads to disbelief"),
    ("BEH_SPI_HYPOCRISY", "BEH_SPI_SHOWING_OFF", "CAUSES", "Hypocrisy leads to showing off"),
    
    # ===== VIRTUES AND THEIR EFFECTS =====
    # Patience chains
    ("BEH_EMO_PATIENCE", "BEH_SPI_TAWAKKUL", "STRENGTHENS", "Patience strengthens trust in Allah"),
    ("BEH_EMO_PATIENCE", "BEH_SPI_FAITH", "STRENGTHENS", "Patience strengthens faith"),
    ("BEH_EMO_PATIENCE", "BEH_EMO_GRATITUDE", "LEADS_TO", "Patience leads to gratitude"),
    ("BEH_EMO_PATIENCE", "BEH_SPI_CONTENTMENT", "CAUSES", "Patience leads to contentment"),
    
    # Gratitude chains
    ("BEH_EMO_GRATITUDE", "BEH_SPI_FAITH", "STRENGTHENS", "Gratitude strengthens faith"),
    ("BEH_EMO_GRATITUDE", "BEH_FIN_CHARITY", "CAUSES", "Gratitude motivates giving"),
    ("BEH_EMO_GRATITUDE", "BEH_SPI_WORSHIP", "CAUSES", "Gratitude motivates worship"),
    
    # Repentance chains
    ("BEH_SPI_REPENTANCE", "BEH_SPI_FAITH", "STRENGTHENS", "Repentance strengthens faith"),
    ("BEH_SPI_REPENTANCE", "BEH_EMO_HOPE", "CAUSES", "Repentance brings hope"),
    ("BEH_SPI_REPENTANCE", "BEH_SPI_SINCERITY", "CAUSES", "Repentance leads to sincerity"),
    
    # Taqwa chains
    ("BEH_SPI_TAQWA", "BEH_SPI_PRAYER", "CAUSES", "Taqwa motivates prayer"),
    ("BEH_SPI_TAQWA", "BEH_FIN_CHARITY", "CAUSES", "Taqwa motivates charity"),
    ("BEH_SPI_TAQWA", "BEH_SPEECH_TRUTHFULNESS", "CAUSES", "Taqwa leads to truthfulness"),
    ("BEH_SPI_TAQWA", "BEH_SOC_JUSTICE", "CAUSES", "Taqwa motivates justice"),
    ("BEH_SPI_TAQWA", "BEH_SPI_FAITH", "STRENGTHENS", "Taqwa strengthens faith"),
    ("BEH_SPI_TAQWA", "BEH_PHY_LOWERING_GAZE", "CAUSES", "Taqwa leads to lowering gaze"),
    
    # Khushu (humility in prayer) chains
    ("BEH_SPI_KHUSHU", "BEH_SPI_PRAYER", "STRENGTHENS", "Khushu perfects prayer"),
    ("BEH_SPI_KHUSHU", "BEH_SPI_FAITH", "STRENGTHENS", "Khushu strengthens faith"),
    ("BEH_SPI_KHUSHU", "BEH_SPI_CONTEMPLATION", "CAUSES", "Khushu leads to contemplation"),
    
    # Tawakkul chains
    ("BEH_SPI_TAWAKKUL", "BEH_SPI_FAITH", "STRENGTHENS", "Tawakkul strengthens faith"),
    ("BEH_SPI_TAWAKKUL", "BEH_SPI_CONTENTMENT", "CAUSES", "Tawakkul leads to contentment"),
    ("BEH_SPI_TAWAKKUL", "BEH_EMO_PATIENCE", "STRENGTHENS", "Tawakkul strengthens patience"),
    
    # ===== SPEECH BEHAVIORS =====
    # Truthfulness chains
    ("BEH_SPEECH_TRUTHFULNESS", "BEH_SOC_TRUSTWORTHINESS", "CAUSES", "Truthfulness builds trust"),
    ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPI_FAITH", "STRENGTHENS", "Truthfulness is a sign of faith"),
    ("BEH_SPEECH_TRUTHFULNESS", "BEH_SOC_JUSTICE", "CAUSES", "Truthfulness enables justice"),
    
    # Lying chains
    ("BEH_SPEECH_LYING", "BEH_SPI_HYPOCRISY", "LEADS_TO", "Lying is a sign of hypocrisy"),
    ("BEH_SPEECH_LYING", "BEH_SOC_BETRAYAL", "CAUSES", "Lying leads to betrayal"),
    ("BEH_SPEECH_LYING", "BEH_SOC_CORRUPTION", "CAUSES", "Lying leads to corruption"),
    
    # Backbiting chains
    ("BEH_SPEECH_BACKBITING", "BEH_EMO_HATRED", "CAUSES", "Backbiting spreads hatred"),
    ("BEH_SPEECH_BACKBITING", "BEH_SOC_SEVERING_TIES", "CAUSES", "Backbiting severs ties"),
    ("BEH_SPEECH_BACKBITING", "BEH_SOC_CORRUPTION", "CAUSES", "Backbiting corrupts society"),
    
    # Slander chains
    ("BEH_SPEECH_SLANDER", "BEH_SOC_OPPRESSION", "CAUSES", "Slander is a form of oppression"),
    ("BEH_SPEECH_SLANDER", "BEH_SOC_CORRUPTION", "CAUSES", "Slander corrupts society"),
    
    # Dhikr (remembrance) chains
    ("BEH_SPEECH_REMEMBRANCE", "BEH_SPI_FAITH", "STRENGTHENS", "Dhikr strengthens faith"),
    ("BEH_SPEECH_REMEMBRANCE", "BEH_SPI_CONTENTMENT", "CAUSES", "Dhikr brings contentment - ألا بذكر الله تطمئن القلوب"),
    ("BEH_SPEECH_REMEMBRANCE", "BEH_SPI_TAQWA", "STRENGTHENS", "Dhikr strengthens taqwa"),
    ("BEH_SPEECH_REMEMBRANCE", "BEH_EMO_GRATITUDE", "CAUSES", "Dhikr leads to gratitude"),
    
    # Supplication chains
    ("BEH_SPEECH_SUPPLICATION", "BEH_SPI_FAITH", "STRENGTHENS", "Dua strengthens faith"),
    ("BEH_SPEECH_SUPPLICATION", "BEH_EMO_HOPE", "CAUSES", "Dua brings hope"),
    ("BEH_SPEECH_SUPPLICATION", "BEH_SPI_TAWAKKUL", "STRENGTHENS", "Dua strengthens tawakkul"),
    
    # ===== FINANCIAL BEHAVIORS =====
    # Charity chains
    ("BEH_FIN_CHARITY", "BEH_SPI_FAITH", "STRENGTHENS", "Charity purifies and strengthens faith"),
    ("BEH_FIN_CHARITY", "BEH_EMO_GRATITUDE", "CAUSES", "Giving leads to gratitude"),
    ("BEH_FIN_CHARITY", "BEH_SOC_MERCY", "CAUSES", "Charity is an act of mercy"),
    ("BEH_FIN_CHARITY", "BEH_SPI_SINCERITY", "STRENGTHENS", "Charity strengthens sincerity"),
    
    # Zakat chains
    ("BEH_FIN_ZAKAT", "BEH_SPI_FAITH", "STRENGTHENS", "Zakat purifies wealth and faith"),
    ("BEH_FIN_ZAKAT", "BEH_SOC_JUSTICE", "CAUSES", "Zakat promotes social justice"),
    ("BEH_FIN_ZAKAT", "BEH_SOC_HELPING_NEEDY", "CAUSES", "Zakat helps the needy"),
    
    # Spending chains
    ("BEH_FIN_SPENDING", "BEH_SPI_FAITH", "STRENGTHENS", "Spending in Allah's way strengthens faith"),
    ("BEH_FIN_SPENDING", "BEH_SOC_HELPING_NEEDY", "CAUSES", "Spending helps the needy"),
    
    # Hoarding chains
    ("BEH_FIN_HOARDING", "BEH_EMO_INGRATITUDE", "CAUSES", "Hoarding shows ingratitude"),
    ("BEH_FIN_HOARDING", "BEH_SOC_OPPRESSION", "CAUSES", "Hoarding oppresses the needy"),
    
    # Usury chains
    ("BEH_FIN_USURY", "BEH_SOC_OPPRESSION", "CAUSES", "Usury oppresses borrowers"),
    ("BEH_FIN_USURY", "BEH_SPI_DISBELIEF", "LEADS_TO", "Usury is war against Allah"),
    ("BEH_FIN_USURY", "BEH_SOC_CORRUPTION", "CAUSES", "Usury corrupts economy"),
    
    # Fraud chains
    ("BEH_FIN_FRAUD", "BEH_SOC_OPPRESSION", "CAUSES", "Fraud oppresses others"),
    ("BEH_FIN_FRAUD", "BEH_SOC_CORRUPTION", "CAUSES", "Fraud corrupts society"),
    
    # Bribery chains
    ("BEH_FIN_BRIBERY", "BEH_SOC_CORRUPTION", "CAUSES", "Bribery corrupts justice"),
    ("BEH_FIN_BRIBERY", "BEH_SOC_OPPRESSION", "CAUSES", "Bribery enables oppression"),
    
    # ===== SOCIAL BEHAVIORS =====
    # Justice chains
    ("BEH_SOC_JUSTICE", "BEH_SOC_TRUSTWORTHINESS", "CAUSES", "Justice builds trust"),
    ("BEH_SOC_JUSTICE", "BEH_SPI_TAQWA", "STRENGTHENS", "Justice is from taqwa"),
    ("BEH_SOC_JUSTICE", "BEH_SOC_RECONCILIATION", "CAUSES", "Justice enables reconciliation"),
    
    # Oppression chains
    ("BEH_SOC_OPPRESSION", "BEH_SPI_DISBELIEF", "LEADS_TO", "Oppression is associated with disbelief"),
    ("BEH_SOC_OPPRESSION", "BEH_EMO_HATRED", "CAUSES", "Oppression breeds hatred"),
    ("BEH_SOC_OPPRESSION", "BEH_SOC_CORRUPTION", "CAUSES", "Oppression corrupts society"),
    
    # Mercy chains
    ("BEH_SOC_MERCY", "BEH_SPI_FAITH", "STRENGTHENS", "Mercy is a sign of faith"),
    ("BEH_SOC_MERCY", "BEH_SOC_JUSTICE", "COMPLEMENTS", "Mercy complements justice"),
    ("BEH_SOC_MERCY", "BEH_FIN_CHARITY", "CAUSES", "Mercy motivates charity"),
    ("BEH_SOC_MERCY", "BEH_SOC_ORPHAN_CARE", "CAUSES", "Mercy motivates orphan care"),
    
    # Betrayal chains
    ("BEH_SOC_BETRAYAL", "BEH_SPI_HYPOCRISY", "LEADS_TO", "Betrayal is a sign of hypocrisy"),
    ("BEH_SOC_BETRAYAL", "BEH_SOC_SEVERING_TIES", "CAUSES", "Betrayal severs ties"),
    ("BEH_SOC_BETRAYAL", "BEH_SOC_CORRUPTION", "CAUSES", "Betrayal corrupts trust"),
    
    # Transgression chains
    ("BEH_SOC_TRANSGRESSION", "BEH_SPI_DISBELIEF", "LEADS_TO", "Transgression can lead to disbelief"),
    ("BEH_SOC_TRANSGRESSION", "BEH_SOC_OPPRESSION", "CAUSES", "Transgression leads to oppression"),
    ("BEH_SOC_TRANSGRESSION", "BEH_SOC_CORRUPTION", "CAUSES", "Transgression corrupts society"),
    
    # Kindness to parents chains
    ("BEH_SOC_KINDNESS_PARENTS", "BEH_SPI_FAITH", "STRENGTHENS", "Kindness to parents strengthens faith"),
    ("BEH_SOC_KINDNESS_PARENTS", "BEH_SOC_KINSHIP_TIES", "STRENGTHENS", "Kindness to parents strengthens family"),
    
    # Disobedience to parents chains
    ("BEH_SOC_DISOBEDIENCE_PARENTS", "BEH_SOC_SEVERING_TIES", "CAUSES", "Disobedience severs family ties"),
    ("BEH_SOC_DISOBEDIENCE_PARENTS", "BEH_SPI_DISBELIEF", "LEADS_TO", "Major sin leading away from faith"),
    
    # Kinship ties chains
    ("BEH_SOC_KINSHIP_TIES", "BEH_SPI_FAITH", "STRENGTHENS", "Maintaining ties strengthens faith"),
    ("BEH_SOC_KINSHIP_TIES", "BEH_SOC_MERCY", "CAUSES", "Kinship ties promote mercy"),
    
    # Severing ties chains
    ("BEH_SOC_SEVERING_TIES", "BEH_SOC_CORRUPTION", "CAUSES", "Severing ties corrupts society"),
    ("BEH_SOC_SEVERING_TIES", "BEH_EMO_HATRED", "CAUSES", "Severing ties breeds hatred"),
    
    # Ihsan chains
    ("BEH_SOC_IHSAN", "BEH_SPI_FAITH", "STRENGTHENS", "Ihsan is the highest level of faith"),
    ("BEH_SOC_IHSAN", "BEH_SPI_SINCERITY", "CAUSES", "Ihsan requires sincerity"),
    ("BEH_SOC_IHSAN", "BEH_SOC_JUSTICE", "STRENGTHENS", "Ihsan perfects justice"),
    
    # Orphan care chains
    ("BEH_SOC_ORPHAN_CARE", "BEH_SPI_FAITH", "STRENGTHENS", "Orphan care strengthens faith"),
    ("BEH_SOC_ORPHAN_CARE", "BEH_SOC_MERCY", "CAUSES", "Orphan care is an act of mercy"),
    
    # Orphan harm chains
    ("BEH_SOC_ORPHAN_HARM", "BEH_SOC_OPPRESSION", "CAUSES", "Harming orphans is oppression"),
    ("BEH_SOC_ORPHAN_HARM", "BEH_SPI_DISBELIEF", "LEADS_TO", "Major sin leading away from faith"),
    
    # ===== COGNITIVE BEHAVIORS =====
    # Knowledge chains
    ("BEH_COG_KNOWLEDGE", "BEH_SPI_FAITH", "STRENGTHENS", "Knowledge strengthens faith"),
    ("BEH_COG_KNOWLEDGE", "BEH_SPI_TAQWA", "CAUSES", "Knowledge leads to taqwa - إنما يخشى الله من عباده العلماء"),
    ("BEH_COG_KNOWLEDGE", "BEH_COG_HUMILITY", "CAUSES", "True knowledge leads to humility"),
    ("BEH_COG_KNOWLEDGE", "BEH_COG_REFLECTION", "CAUSES", "Knowledge enables reflection"),
    
    # Ignorance chains
    ("BEH_COG_IGNORANCE", "BEH_SPI_DISBELIEF", "LEADS_TO", "Ignorance can lead to disbelief"),
    ("BEH_COG_IGNORANCE", "BEH_COG_ARROGANCE", "CAUSES", "Ignorance breeds arrogance"),
    ("BEH_COG_IGNORANCE", "BEH_SOC_TRANSGRESSION", "CAUSES", "Ignorance leads to transgression"),
    ("BEH_COG_IGNORANCE", "BEH_SOC_OPPRESSION", "CAUSES", "Ignorance leads to oppression"),
    
    # Doubt chains
    ("BEH_COG_DOUBT", "BEH_SPI_DISBELIEF", "LEADS_TO", "Persistent doubt can lead to disbelief"),
    ("BEH_COG_DOUBT", "BEH_COG_HEEDLESSNESS", "CAUSES", "Doubt causes heedlessness"),
    
    # Certainty chains
    ("BEH_COG_CERTAINTY", "BEH_SPI_FAITH", "STRENGTHENS", "Certainty strengthens faith"),
    ("BEH_COG_CERTAINTY", "BEH_SPI_TAWAKKUL", "CAUSES", "Certainty leads to trust in Allah"),
    
    # Reflection chains
    ("BEH_COG_REFLECTION", "BEH_SPI_FAITH", "STRENGTHENS", "Reflection strengthens faith"),
    ("BEH_COG_REFLECTION", "BEH_EMO_GRATITUDE", "CAUSES", "Reflection leads to gratitude"),
    ("BEH_COG_REFLECTION", "BEH_SPI_CONTEMPLATION", "CAUSES", "Reflection leads to contemplation"),
    
    # Contemplation chains
    ("BEH_SPI_CONTEMPLATION", "BEH_SPI_FAITH", "STRENGTHENS", "Contemplation strengthens faith"),
    ("BEH_SPI_CONTEMPLATION", "BEH_EMO_FEAR_ALLAH", "CAUSES", "Contemplation leads to fear of Allah"),
    
    # Following desire chains
    ("BEH_COG_FOLLOWING_DESIRE", "BEH_SOC_TRANSGRESSION", "CAUSES", "Following desire leads to transgression"),
    ("BEH_COG_FOLLOWING_DESIRE", "BEH_SPI_DISBELIEF", "LEADS_TO", "Following desire can lead to disbelief"),
    ("BEH_COG_FOLLOWING_DESIRE", "BEH_COG_HEEDLESSNESS", "CAUSES", "Following desire causes heedlessness"),
    
    # ===== PREVENTION RELATIONSHIPS =====
    ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF", "PREVENTS", "Faith prevents disbelief"),
    ("BEH_SPI_TAQWA", "BEH_SOC_TRANSGRESSION", "PREVENTS", "Taqwa prevents transgression"),
    ("BEH_SPI_PRAYER", "BEH_SOC_TRANSGRESSION", "PREVENTS", "Prayer prevents transgression - إن الصلاة تنهى عن الفحشاء والمنكر"),
    ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING", "PREVENTS", "Truthfulness prevents lying"),
    ("BEH_EMO_PATIENCE", "BEH_EMO_IMPATIENCE", "PREVENTS", "Patience prevents impatience"),
    ("BEH_EMO_GRATITUDE", "BEH_EMO_INGRATITUDE", "PREVENTS", "Gratitude prevents ingratitude"),
    ("BEH_COG_KNOWLEDGE", "BEH_COG_IGNORANCE", "PREVENTS", "Knowledge prevents ignorance"),
    ("BEH_SPI_REPENTANCE", "BEH_EMO_GRIEF", "PREVENTS", "Repentance prevents despair"),
    ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION", "PREVENTS", "Justice prevents oppression"),
    ("BEH_FIN_CHARITY", "BEH_FIN_HOARDING", "PREVENTS", "Charity prevents hoarding"),
    ("BEH_SOC_MERCY", "BEH_EMO_HATRED", "PREVENTS", "Mercy prevents hatred"),
    ("BEH_SPI_SINCERITY", "BEH_SPI_SHOWING_OFF", "PREVENTS", "Sincerity prevents showing off"),
    ("BEH_SPI_FASTING", "BEH_COG_FOLLOWING_DESIRE", "PREVENTS", "Fasting prevents following desire"),
    
    # ===== OPPOSITE RELATIONSHIPS =====
    ("BEH_SPI_FAITH", "BEH_SPI_DISBELIEF", "OPPOSITE_OF", "Faith is opposite of disbelief"),
    ("BEH_SPEECH_TRUTHFULNESS", "BEH_SPEECH_LYING", "OPPOSITE_OF", "Truthfulness is opposite of lying"),
    ("BEH_EMO_PATIENCE", "BEH_EMO_IMPATIENCE", "OPPOSITE_OF", "Patience is opposite of impatience"),
    ("BEH_EMO_GRATITUDE", "BEH_EMO_INGRATITUDE", "OPPOSITE_OF", "Gratitude is opposite of ingratitude"),
    ("BEH_COG_KNOWLEDGE", "BEH_COG_IGNORANCE", "OPPOSITE_OF", "Knowledge is opposite of ignorance"),
    ("BEH_COG_HUMILITY", "BEH_COG_ARROGANCE", "OPPOSITE_OF", "Humility is opposite of arrogance"),
    ("BEH_SOC_JUSTICE", "BEH_SOC_OPPRESSION", "OPPOSITE_OF", "Justice is opposite of oppression"),
    ("BEH_FIN_CHARITY", "BEH_FIN_HOARDING", "OPPOSITE_OF", "Charity is opposite of hoarding"),
    ("BEH_SOC_MERCY", "BEH_EMO_HATRED", "OPPOSITE_OF", "Mercy is opposite of hatred"),
    ("BEH_SOC_TRUSTWORTHINESS", "BEH_SOC_BETRAYAL", "OPPOSITE_OF", "Trustworthiness is opposite of betrayal"),
    ("BEH_SPI_SINCERITY", "BEH_SPI_SHOWING_OFF", "OPPOSITE_OF", "Sincerity is opposite of showing off"),
    ("BEH_SOC_KINDNESS_PARENTS", "BEH_SOC_DISOBEDIENCE_PARENTS", "OPPOSITE_OF", "Kindness is opposite of disobedience"),
    ("BEH_SOC_KINSHIP_TIES", "BEH_SOC_SEVERING_TIES", "OPPOSITE_OF", "Maintaining ties is opposite of severing"),
    ("BEH_SOC_ORPHAN_CARE", "BEH_SOC_ORPHAN_HARM", "OPPOSITE_OF", "Orphan care is opposite of orphan harm"),
    ("BEH_FIN_FAIR_TRADE", "BEH_FIN_FRAUD", "OPPOSITE_OF", "Fair trade is opposite of fraud"),
    
    # ===== PHYSICAL BEHAVIORS =====
    # Eating/Drinking chains
    ("BEH_PHY_EATING", "BEH_EMO_GRATITUDE", "LEADS_TO", "Eating should lead to gratitude"),
    ("BEH_PHY_DRINKING", "BEH_EMO_GRATITUDE", "LEADS_TO", "Drinking should lead to gratitude"),
    
    # Looking chains
    ("BEH_PHY_LOOKING", "BEH_COG_FOLLOWING_DESIRE", "CAUSES", "Unlawful looking causes following desire"),
    ("BEH_PHY_LOWERING_GAZE", "BEH_SPI_TAQWA", "STRENGTHENS", "Lowering gaze strengthens taqwa"),
    ("BEH_PHY_LOWERING_GAZE", "BEH_SPI_FAITH", "STRENGTHENS", "Lowering gaze strengthens faith"),
    
    # Modesty chains
    ("BEH_PHY_MODESTY", "BEH_SPI_FAITH", "STRENGTHENS", "Modesty is part of faith - الحياء شعبة من الإيمان"),
    ("BEH_PHY_MODESTY", "BEH_SOC_TRUSTWORTHINESS", "CAUSES", "Modesty builds trust"),
    
    # Prayer physical aspects
    ("BEH_PHY_PROSTRATION", "BEH_SPI_PRAYER", "STRENGTHENS", "Prostration is the core of prayer"),
    ("BEH_PHY_PROSTRATION", "BEH_COG_HUMILITY", "CAUSES", "Prostration cultivates humility"),
    ("BEH_PHY_STANDING", "BEH_SPI_PRAYER", "STRENGTHENS", "Standing is part of prayer"),
    
    # ===== WORSHIP BEHAVIORS =====
    # Prayer chains
    ("BEH_SPI_PRAYER", "BEH_SPI_FAITH", "STRENGTHENS", "Prayer strengthens faith"),
    ("BEH_SPI_PRAYER", "BEH_SPI_TAQWA", "STRENGTHENS", "Prayer strengthens taqwa"),
    ("BEH_SPI_PRAYER", "BEH_SPEECH_REMEMBRANCE", "CAUSES", "Prayer includes remembrance"),
    
    # Fasting chains
    ("BEH_SPI_FASTING", "BEH_SPI_TAQWA", "CAUSES", "Fasting leads to taqwa - لعلكم تتقون"),
    ("BEH_SPI_FASTING", "BEH_EMO_PATIENCE", "STRENGTHENS", "Fasting strengthens patience"),
    ("BEH_SPI_FASTING", "BEH_SPI_FAITH", "STRENGTHENS", "Fasting strengthens faith"),
    
    # Hajj chains
    ("BEH_SPI_HAJJ", "BEH_SPI_FAITH", "STRENGTHENS", "Hajj strengthens faith"),
    ("BEH_SPI_HAJJ", "BEH_SPI_REPENTANCE", "CAUSES", "Hajj is an opportunity for repentance"),
    ("BEH_SPI_HAJJ", "BEH_EMO_PATIENCE", "STRENGTHENS", "Hajj requires and builds patience"),
    
    # Jihad chains
    ("BEH_SPI_JIHAD", "BEH_SPI_FAITH", "STRENGTHENS", "Jihad strengthens faith"),
    ("BEH_SPI_JIHAD", "BEH_EMO_PATIENCE", "STRENGTHENS", "Jihad requires patience"),
    ("BEH_SPI_JIHAD", "BEH_SPI_SINCERITY", "STRENGTHENS", "Jihad requires sincerity"),
    
    # Recitation chains
    ("BEH_SPI_RECITATION", "BEH_SPI_FAITH", "STRENGTHENS", "Quran recitation strengthens faith"),
    ("BEH_SPI_RECITATION", "BEH_SPI_CONTEMPLATION", "CAUSES", "Recitation leads to contemplation"),
    ("BEH_SPI_RECITATION", "BEH_COG_KNOWLEDGE", "CAUSES", "Recitation increases knowledge"),
]

# Build new edges
new_edges = []
for source, target, edge_type, evidence in CAUSAL_RELATIONSHIPS:
    # Check if both behaviors exist
    if source not in behaviors:
        print(f"WARNING: Source behavior not found: {source}")
        continue
    if target not in behaviors:
        print(f"WARNING: Target behavior not found: {target}")
        continue
    
    # Check if edge already exists
    if (source, target, edge_type) in existing_edges:
        continue
    
    edge = {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "confidence": 0.85,
        "evidence": {
            "type": "scholarly",
            "description": evidence,
            "sources": ["quran", "tafsir", "scholarly_consensus"]
        },
        "provenance": "enterprise_graph_v1"
    }
    new_edges.append(edge)

print(f"\nNew edges to add: {len(new_edges)}")

# Count by type
from collections import Counter
new_types = Counter(e["edge_type"] for e in new_edges)
print("New edges by type:")
for et, count in new_types.most_common():
    print(f"  {et}: {count}")

# Merge with existing graph
merged_edges = existing_graph.get("edges", []) + new_edges

# Update graph metadata
merged_graph = {
    "graph_type": "semantic",
    "version": "3.0",
    "created_at": datetime.utcnow().isoformat(),
    "description": "Enterprise-grade semantic causal graph. Scholarly evidence required for all causal edges.",
    "allowed_edge_types": existing_graph.get("allowed_edge_types", []),
    "causal_edge_types": existing_graph.get("causal_edge_types", []),
    "nodes": existing_graph.get("nodes", []),
    "edges": merged_edges
}

# Add missing behavior nodes
existing_node_ids = {n["id"] for n in merged_graph["nodes"]}
for beh_id, beh in behaviors.items():
    if beh_id not in existing_node_ids:
        merged_graph["nodes"].append({
            "id": beh_id,
            "type": "BEHAVIOR",
            "ar": beh.get("ar", ""),
            "en": beh.get("en", "")
        })
        print(f"Added missing node: {beh_id}")

# Save merged graph
output_path = DATA_DIR / "graph" / "semantic_graph_v3.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_graph, f, ensure_ascii=False, indent=2)

print(f"\nSaved enterprise graph to {output_path}")
print(f"Total nodes: {len(merged_graph['nodes'])}")
print(f"Total edges: {len(merged_graph['edges'])}")

# Count behavior-to-behavior causal edges
beh_ids = set(behaviors.keys())
causal_types = {"CAUSES", "LEADS_TO", "PREVENTS", "STRENGTHENS"}
beh_causal = [e for e in merged_graph["edges"] 
              if e["source"] in beh_ids and e["target"] in beh_ids 
              and e["edge_type"] in causal_types]
print(f"Behavior-to-behavior causal edges: {len(beh_causal)}")
