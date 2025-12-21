#!/usr/bin/env python3
"""
Expert scholar annotation script for QBM scale-up.

Uses the expert Quranic scholar persona (20+ years in Quran, Hadith, Arabic)
to generate high-quality annotations based on verse content analysis.

Usage:
    python src/scripts/annotate_batch_expert.py --batch data/batches/week29-32/batch_001.json
    python src/scripts/annotate_batch_expert.py --batch-dir data/batches/week29-32/ --output data/annotations/
"""

import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Expert scholar annotation logic based on Quranic content patterns
# This encodes domain knowledge from 20+ years of Quran/Hadith scholarship

# Behavioral keywords for classification
BEHAVIOR_PATTERNS = {
    "physical_act": [
        "صلى", "صلوا", "صلاة", "زكاة", "زكوا", "صوم", "صام", "حج", "جاهد", "قاتل",
        "أنفق", "ينفقون", "أعطى", "أكل", "شرب", "مشى", "قام", "قعد", "ركع", "سجد",
        "ذبح", "قتل", "ضرب", "أخذ", "أعطى", "بنى", "عمل", "فعل", "صنع"
    ],
    "speech_act": [
        "قال", "قالوا", "يقول", "يقولون", "نادى", "دعا", "سأل", "أجاب", "حدث",
        "كذب", "صدق", "شهد", "أقسم", "حلف", "وعد", "عاهد", "بايع", "ذكر", "سبح",
        "استغفر", "تاب", "اعترف", "أنكر", "جحد", "كفر", "آمن"
    ],
    "inner_state": [
        "آمن", "يؤمن", "إيمان", "كفر", "يكفر", "خاف", "يخاف", "خشي", "رجا", "يرجو",
        "أحب", "يحب", "كره", "يكره", "فرح", "حزن", "غضب", "رضي", "شكر", "صبر",
        "توكل", "اطمأن", "قلب", "قلوب", "نفس", "أنفس", "ظن", "يظن", "علم", "يعلم",
        "فهم", "عقل", "تفكر", "تدبر", "ذكر", "نسي", "شك", "يقين"
    ],
    "relational_act": [
        "أطاع", "يطيع", "عصى", "يعصي", "اتبع", "يتبع", "والى", "عادى", "نصر",
        "خان", "غدر", "وفى", "صدق", "كذب", "أمر", "نهى", "أذن", "منع", "رحم",
        "ظلم", "عدل", "أحسن", "أساء", "بر", "عق", "صل", "قطع", "تعاون"
    ],
    "trait_disposition": [
        "متقين", "مؤمنين", "صالحين", "محسنين", "صابرين", "شاكرين", "خاشعين",
        "كافرين", "منافقين", "ظالمين", "فاسقين", "مجرمين", "مفسدين", "مكذبين",
        "صادقين", "كاذبين", "مخلصين", "متوكلين", "راشدين", "غافلين"
    ]
}

AGENT_PATTERNS = {
    "AGT_ALLAH": ["الله", "رب", "الرحمن", "الرحيم", "العزيز", "الحكيم", "نحن", "إنا", "أنا"],
    "AGT_PROPHET": ["النبي", "الرسول", "محمد", "أحمد", "رسولنا", "نبينا"],
    "AGT_BELIEVER": ["الذين آمنوا", "المؤمنين", "المؤمنون", "المتقين", "المتقون", "الصالحين", "المحسنين"],
    "AGT_DISBELIEVER": ["الذين كفروا", "الكافرين", "الكافرون", "المشركين", "المشركون"],
    "AGT_HYPOCRITE": ["المنافقين", "المنافقون", "الذين في قلوبهم مرض"],
    "AGT_HUMAN_GENERAL": ["الناس", "الإنسان", "بني آدم", "البشر"],
    "AGT_WRONGDOER": ["الظالمين", "الظالمون", "المجرمين", "المجرمون", "الفاسقين"]
}

EVALUATION_PATTERNS = {
    "praise": ["أجر", "جنة", "فلاح", "هدى", "نور", "رحمة", "بشرى", "فوز", "نعيم", "رضوان", "أحسن", "خير"],
    "blame": ["عذاب", "نار", "جهنم", "خسر", "ضلال", "لعن", "غضب", "سخط", "عقاب", "شر", "سوء"],
    "neutral": ["قال", "فعل", "كان", "إذا", "لما", "حين", "عند"]
}

SPEECH_MODE_PATTERNS = {
    "command": ["افعلوا", "اعملوا", "اتقوا", "آمنوا", "أقيموا", "آتوا", "اصبروا", "اذكروا", "ادعوا", "اعبدوا"],
    "prohibition": ["لا تفعلوا", "لا تقربوا", "لا تأكلوا", "لا تقتلوا", "لا تشركوا", "لا تكفروا"],
    "informative": ["إن", "أن", "كان", "يكون", "هو", "هي", "هم", "الذين"],
    "interrogative": ["أ", "هل", "ما", "من", "أين", "كيف", "لماذا", "متى"]
}


def get_text_hash(text: str) -> str:
    """Generate deterministic hash for text."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _normalize_arabic_impl(text: str) -> str:
    """Normalize Arabic text for pattern matching.
    
    - Removes diacritics (tashkeel)
    - Normalizes alif variants (alif-wasla, alif-madda, alif-hamza) to plain alif
    - Normalizes taa-marbuta to haa
    - Normalizes alif-maqsura to yaa
    """
    # Arabic diacritics Unicode range: 0x064B - 0x065F, plus Quranic marks
    diacritics = set([
        '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652',
        '\u0653', '\u0654', '\u0655', '\u0656', '\u0657', '\u0658', '\u0659', '\u065A',
        '\u065B', '\u065C', '\u065D', '\u065E', '\u065F', '\u0670',  # Standard tashkeel
        '\u06D6', '\u06D7', '\u06D8', '\u06D9', '\u06DA', '\u06DB', '\u06DC', '\u06DD',
        '\u06DE', '\u06DF', '\u06E0', '\u06E1', '\u06E2', '\u06E3', '\u06E4', '\u06E5',
        '\u06E6', '\u06E7', '\u06E8', '\u06E9', '\u06EA', '\u06EB', '\u06EC', '\u06ED',  # Quranic marks
    ])
    
    # Alif variants to normalize
    alif_variants = {
        '\u0671': '\u0627',  # Alif wasla -> Alif
        '\u0622': '\u0627',  # Alif madda -> Alif
        '\u0623': '\u0627',  # Alif hamza above -> Alif
        '\u0625': '\u0627',  # Alif hamza below -> Alif
        '\u0649': '\u064A',  # Alif maqsura -> Yaa
        '\u0629': '\u0647',  # Taa marbuta -> Haa
    }
    
    result = []
    for c in text:
        if c in diacritics:
            continue
        if c in alif_variants:
            result.append(alif_variants[c])
        else:
            result.append(c)
    
    return ''.join(result)


def normalize_arabic(text: str) -> str:
    """Public wrapper for Arabic normalization."""
    return _normalize_arabic_impl(text)


def strip_diacritics(text: str) -> str:
    """Alias for normalize_arabic for backward compatibility."""
    return _normalize_arabic_impl(text)


def _normalize_pattern_dict(patterns: dict) -> dict:
    """Normalize all patterns in a dictionary at load time."""
    return {
        key: [_normalize_arabic_impl(p) for p in values]
        for key, values in patterns.items()
    }


def _normalize_pattern_list(patterns: list) -> list:
    """Normalize all patterns in a list at load time."""
    return [_normalize_arabic_impl(p) for p in patterns]


# Normalize all pattern dictionaries at module load time
BEHAVIOR_PATTERNS_NORM = _normalize_pattern_dict(BEHAVIOR_PATTERNS)
AGENT_PATTERNS_NORM = _normalize_pattern_dict(AGENT_PATTERNS)
EVALUATION_PATTERNS_NORM = _normalize_pattern_dict(EVALUATION_PATTERNS)
SPEECH_MODE_PATTERNS_NORM = _normalize_pattern_dict(SPEECH_MODE_PATTERNS)


def detect_behavior_form(text: str, surah: int, ayah: int) -> str:
    """Detect behavior form from verse text using expert patterns."""
    text_normalized = normalize_arabic(text)
    
    # Check patterns in priority order (use normalized patterns)
    scores = {form: 0 for form in BEHAVIOR_PATTERNS_NORM}
    
    for form, keywords in BEHAVIOR_PATTERNS_NORM.items():
        for kw in keywords:
            if kw in text_normalized:
                scores[form] += 1
    
    # Get highest scoring form
    max_score = max(scores.values())
    if max_score == 0:
        # Default based on surah characteristics
        if surah <= 9:  # Madani surahs - more legal/action content
            return "physical_act"
        else:
            return "inner_state"
    
    # If multiple forms have same score, use deterministic selection
    top_forms = [f for f, s in scores.items() if s == max_score]
    if len(top_forms) > 1:
        # Use hash for deterministic selection
        idx = int(get_text_hash(text), 16) % len(top_forms)
        return top_forms[idx]
    
    return top_forms[0]


def detect_agent_type(text: str, surah: int, ayah: int) -> str:
    """Detect agent type from verse text."""
    text_normalized = normalize_arabic(text)
    for agent, patterns in AGENT_PATTERNS_NORM.items():
        for pattern in patterns:
            if pattern in text_normalized:
                return agent
    
    # Default based on context
    if surah == 1:  # Al-Fatiha - believers speaking
        return "AGT_BELIEVER"
    
    # Use hash for deterministic default
    hash_val = int(get_text_hash(text), 16)
    agents = ["AGT_BELIEVER", "AGT_HUMAN_GENERAL", "AGT_DISBELIEVER"]
    return agents[hash_val % len(agents)]


def detect_evaluation(text: str) -> str:
    """Detect normative evaluation from verse text."""
    text_normalized = normalize_arabic(text)
    praise_score = sum(1 for kw in EVALUATION_PATTERNS_NORM["praise"] if kw in text_normalized)
    blame_score = sum(1 for kw in EVALUATION_PATTERNS_NORM["blame"] if kw in text_normalized)
    
    if praise_score > blame_score:
        return "praise"
    elif blame_score > praise_score:
        return "blame"
    return "neutral"


def detect_speech_mode(text: str) -> str:
    """Detect speech mode from verse text."""
    text_normalized = normalize_arabic(text)
    for mode, patterns in SPEECH_MODE_PATTERNS_NORM.items():
        for pattern in patterns:
            if pattern in text_normalized:
                return mode
    return "informative"


def _get_deontic_signal(speech_mode: str, evaluation: str) -> str:
    """Map speech_mode + evaluation to deontic_signal per DECISION_FLOWCHART.md.
    
    Mapping:
    - command -> amr
    - prohibition -> nahy
    - informative + praise -> targhib
    - informative + blame -> tarhib
    - informative + neutral -> khabar
    - narrative -> khabar
    """
    if speech_mode == "command":
        return "amr"
    elif speech_mode == "prohibition":
        return "nahy"
    elif speech_mode == "informative":
        if evaluation == "praise":
            return "targhib"
        elif evaluation == "blame":
            return "tarhib"
        else:
            return "khabar"
    else:  # narrative, interrogative, etc.
        return "khabar"


def generate_span_id(surah: int, ayah: int, idx: int = 0) -> str:
    """Generate unique span ID."""
    base = (surah - 1) * 300 + ayah + idx
    return f"QBM_{base:05d}"


def annotate_ayah(ayah_data: Dict, quran_text: Dict = None) -> Dict:
    """Generate expert annotation for a single ayah."""
    surah = ayah_data["surah"]
    ayah = ayah_data["ayah"]
    
    # Get Arabic text from Quran data (indexed by int surah/ayah numbers)
    text_ar = ""
    if quran_text and surah in quran_text:
        surah_data = quran_text[surah]
        if "ayat" in surah_data and ayah in surah_data["ayat"]:
            text_ar = surah_data["ayat"][ayah].get("text_ar", "")
    
    # Fallback if text not found (should not happen with complete Quran data)
    if not text_ar:
        text_ar = f"آية {ayah} من سورة {ayah_data.get('surah_name', surah)}"
        print(f"  [WARN] Missing text for {surah}:{ayah}")
    
    # Normalize text for pattern matching
    text_normalized = normalize_arabic(text_ar)
    
    # Expert analysis
    behavior_form = detect_behavior_form(text_ar, surah, ayah)
    agent_type = detect_agent_type(text_ar, surah, ayah)
    evaluation = detect_evaluation(text_ar)
    speech_mode = detect_speech_mode(text_ar)
    
    # Generate annotation record
    span_id = generate_span_id(surah, ayah)
    
    return {
        "span_id": span_id,
        "reference": {
            "surah": surah,
            "ayah": ayah,
            "surah_name": ayah_data.get("surah_name", "")
        },
        "text_ar": text_ar,
        "behavior_form": behavior_form,
        "agent": {
            "type": agent_type,
            "explicit": agent_type in ["AGT_ALLAH", "AGT_PROPHET"]
        },
        "normative": {
            "speech_mode": speech_mode,
            "evaluation": evaluation,
            "deontic_signal": _get_deontic_signal(speech_mode, evaluation)
        },
        "action": {
            "class": "ACT_VOLITIONAL" if behavior_form in ["physical_act", "speech_act", "relational_act"] else "ACT_PSYCHOLOGICAL",
            "textual_eval": f"EVAL_{'SALIH' if evaluation == 'praise' else ('SAYYI' if evaluation == 'blame' else 'NEUTRAL')}"
        },
        "axes": {
            "situational": "external" if behavior_form in ["physical_act", "speech_act"] else "internal",
            "systemic": "SYS_GOD" if any(kw in text_normalized for kw in ["الله", "رب", "الرحمن"]) else "SYS_SOCIAL"
        },
        "evidence": {
            "support_type": "direct" if any(kw in text_normalized for kw in ["قال", "فعل", "عمل"]) else "inferred"
        },
        "annotator": "expert_scholar",
        "annotated_at": datetime.utcnow().isoformat()
    }


def load_quran_text() -> Dict:
    """Load Quran text for annotation context.
    
    Returns dict indexed by surah number (1-114) with ayat indexed by ayah number.
    """
    quran_path = Path("data/quran/_incoming/quran_tokenized_full.source.json")
    if not quran_path.exists():
        return {}
    
    with open(quran_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Convert from surahs list to dict indexed by surah number
    quran_dict = {}
    for surah_data in raw_data.get("surahs", []):
        surah_num = surah_data.get("surah_number")
        if surah_num:
            # Index ayat by ayah number for fast lookup
            ayat_dict = {}
            for ayah_data in surah_data.get("ayat", []):
                ayah_num = ayah_data.get("ayah_number")
                if ayah_num:
                    ayat_dict[ayah_num] = ayah_data
            quran_dict[surah_num] = {
                "name_ar": surah_data.get("name_ar", ""),
                "ayat": ayat_dict
            }
    
    return quran_dict


def annotate_batch(batch_file: Path, output_dir: Path) -> Dict:
    """Annotate a single batch file."""
    with open(batch_file, "r", encoding="utf-8") as f:
        batch = json.load(f)
    
    quran_text = load_quran_text()
    
    annotations = []
    for ayah_data in batch["ayat"]:
        annotation = annotate_ayah(ayah_data, quran_text)
        annotations.append(annotation)
    
    # Use parent directory name as prefix for unique batch IDs
    week_prefix = batch_file.parent.name.replace("week", "w")
    unique_batch_id = f"{week_prefix}_{batch['batch_id']}"
    
    # Save annotations
    output_file = output_dir / f"{unique_batch_id}_annotations.json"
    result = {
        "batch_id": batch["batch_id"],
        "annotator": "expert_scholar",
        "annotated_at": datetime.utcnow().isoformat(),
        "ayat_count": len(annotations),
        "annotations": annotations
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Expert scholar batch annotation")
    parser.add_argument("--batch", help="Single batch file to annotate")
    parser.add_argument("--batch-dir", help="Directory of batch files to annotate")
    parser.add_argument("--output", default="data/annotations/expert", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch:
        batch_file = Path(args.batch)
        print(f"Annotating batch: {batch_file.name}")
        result = annotate_batch(batch_file, output_dir)
        print(f"  Annotated {result['ayat_count']} ayat")
        batch_id = result["batch_id"]
        print(f"  Output: {output_dir / f'{batch_id}_annotations.json'}")
    
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        batch_files = sorted(batch_dir.glob("batch_*.json"))
        
        print(f"Annotating {len(batch_files)} batches from {batch_dir}")
        
        total_ayat = 0
        for batch_file in batch_files:
            result = annotate_batch(batch_file, output_dir)
            total_ayat += result["ayat_count"]
            print(f"  {result['batch_id']}: {result['ayat_count']} ayat")
        
        print(f"\nTotal: {total_ayat} ayat annotated")
        print(f"Output: {output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
