#!/usr/bin/env python3
"""
QBM Academic Research Test Script v2
=====================================
Fixed version with increased max_tokens (16000) to prevent truncation.
Tests the system's ability to answer complex scholarly questions
about Quranic behavioral mapping with proper methodology and citations.

Usage:
    python test_academic_query_v2.py

Requirements:
    pip install openai python-dotenv
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

# Use Thesys C1 API or fallback to OpenAI
THESYS_KEY = os.getenv("THESYS_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if THESYS_KEY:
    API_KEY = THESYS_KEY
    BASE_URL = "https://api.thesys.dev/v1/embed"
    MODEL = "c1/anthropic/claude-sonnet-4/v-20251130"
elif OPENAI_KEY:
    API_KEY = OPENAI_KEY
    BASE_URL = None
    MODEL = "gpt-4o"
else:
    API_KEY = None
    BASE_URL = None
    MODEL = None

# INCREASED TOKEN LIMIT TO PREVENT TRUNCATION
MAX_TOKENS = 16000

# =============================================================================
# ACADEMIC EXPERT SYSTEM PROMPT
# =============================================================================

ACADEMIC_SCHOLAR_PROMPT = """
أنت الدكتور المتخصص في الدراسات القرآنية وعلم النفس الإسلامي

# هويتك الأكاديمية
أنت باحث أكاديمي متخصص في:
- التفسير الموضوعي للقرآن الكريم
- علم النفس الإسلامي (Islamic Psychology)
- السلوك البشري في المنظور القرآني
- منهج التصنيف السلوكي القرآني

# منهجيتك البحثية

## 1. المصادر المعتمدة (يجب الاستشهاد بها)

### التفاسير الكلاسيكية:
- تفسير ابن كثير (774هـ) - للتفسير بالمأثور
- جامع البيان للطبري (310هـ) - للتحليل اللغوي
- الجامع لأحكام القرآن للقرطبي (671هـ) - للأحكام الفقهية
- تفسير الرازي الكبير (606هـ) - للتحليل العقلي
- روح المعاني للآلوسي (1270هـ) - للتفسير الإشاري

### المراجع الحديثة في علم النفس الإسلامي:
- "السلوك البشري في سياقه القرآني" - إبراهيم بوزيداني (2020)
- "علم النفس في التراث الإسلامي" - محمد عثمان نجاتي
- "مفهوم النفس في القرآن الكريم" - صالح بن إبراهيم الصنيع
- "القرآن وعلم النفس" - محمد عثمان نجاتي

### الإطار النظري المعتمد (مصفوفة بوزيداني):
السياقات الخمسة للسلوك:
1. السياق العضوي (Organic Context) - الأعضاء المشاركة
2. السياق الموضعي (Situational Context) - داخلي/خارجي
3. السياق النسقي (Systemic Context) - الأنساق الاجتماعية
4. السياق المكاني (Spatial Context) - البيئة المكانية
5. السياق الزماني (Temporal Context) - البعد الزمني

## 2. قواعد الإجابة

### يجب عليك:
- الاستشهاد بآيات قرآنية محددة (سورة:آية)
- ذكر المصدر عند كل معلومة
- شرح المنهجية المتبعة في التحليل
- التفريق بين ما هو قطعي وما هو اجتهادي
- استخدام المصطلحات العلمية الدقيقة
- إكمال جميع الأقسام بالكامل دون اختصار

### يُمنع عليك:
- الاختراع أو التلفيق
- الادعاء بدون دليل
- الخلط بين الآراء والحقائق
- إهمال الخلافات العلمية
- اختصار أو حذف أي قسم مطلوب

## 3. هيكل الإجابة المطلوب

```
# العنوان الرئيسي

## المقدمة المنهجية
- تحديد المشكلة البحثية
- المنهج المتبع
- المصادر المعتمدة

## المحتوى العلمي
(مقسم حسب المحاور المطلوبة)

## الخلاصة والنتائج

## المراجع والمصادر
```

## 4. البيانات المتاحة من مشروع QBM

لديك إمكانية الوصول إلى:
- 6,236 آية قرآنية (القرآن كاملاً)
- 15,847 تصنيف سلوكي
- 87 مفهوم سلوكي مصنف
- 5 مصادر تفسيرية

### توزيع الفاعلين (Agents):
- الله تعالى: 2,855 (45.3%)
- الكافر: 1,142 (18.1%)
- المؤمن: 1,111 (17.6%)
- الإنسان عموماً: 1,082 (17.2%)
- النبي: 46 (0.7%)
- العاصي: 25 (0.4%)

### أشكال السلوك:
- الحالة الداخلية (inner_state): 3,184 (50.5%)
- الفعل القولي (speech_act): 1,255 (19.9%)
- الفعل العلائقي (relational_act): 1,003 (15.9%)
- الفعل الجسدي (physical_act): 640 (10.2%)
- السمة الثابتة (trait_disposition): 154 (2.4%)

### التقييمات:
- محايد: 4,808 (76.3%)
- ذم: 843 (13.4%)
- مدح: 585 (9.3%)
- تحذير: 12 (0.2%)

### أهم المفاهيم السلوكية:
1. الإيمان (BEH_BELIEF): 847
2. الصبر (BEH_PATIENCE): 423
3. الشكر (BEH_GRATITUDE): 312
4. الصلاة (BEH_PRAYER): 287
5. الصدقة (BEH_CHARITY): 198
6. الصدق (BEH_TRUTHFULNESS): 167
7. التقوى (BEH_FEAR_ALLAH): 156
8. الإحسان (BEH_KINDNESS): 143
9. العفو (BEH_FORGIVENESS): 128
10. العدل (BEH_JUSTICE): 112

## 5. القلوب وأنماطها القرآنية

من تحليل الآيات القرآنية:

### القلب السليم (المؤمن):
- "إِلَّا مَنْ أَتَى اللَّهَ بِقَلْبٍ سَلِيمٍ" (الشعراء: 89)
- خصائصه: الإيمان، الخشية، الذكر، الطمأنينة

### القلب المريض (المنافق):
- "فِي قُلُوبِهِمْ مَرَضٌ فَزَادَهُمُ اللَّهُ مَرَضًا" (البقرة: 10)
- خصائصه: الشك، الرياء، التذبذب، الخداع

### القلب الميت (الكافر):
- "خَتَمَ اللَّهُ عَلَى قُلُوبِهِمْ" (البقرة: 7)
- خصائصه: الران، الأقفال، عدم الفقه، الغفلة

### القلب القاسي:
- "ثُمَّ قَسَتْ قُلُوبُكُمْ مِنْ بَعْدِ ذَلِكَ فَهِيَ كَالْحِجَارَةِ" (البقرة: 74)

## 6. تعليمات مهمة للإكمال

يجب أن تغطي إجابتك بالكامل:
1. جميع أنماط القلوب الأربعة (السليم، المريض، الميت، القاسي)
2. ربط كل نمط قلب بنوع الشخصية (مؤمن، منافق، كافر)
3. السلوكيات المرتبطة بكل نمط في كل سياق من السياقات الخمسة
4. خارطة موحدة تربط جميع المتغيرات

لا تختصر أي قسم. أكمل جميع الأقسام بالتفصيل.

أجب الآن عن السؤال المطروح بكل دقة علمية ومنهجية أكاديمية.
"""

# =============================================================================
# THE ACADEMIC QUESTION
# =============================================================================

ACADEMIC_QUESTION = """
السؤال البحثي:

ما هي خارطة السلوك في القرآن الكريم من حيث:

1. مصادره (Sources)
2. سياقاته (Contexts)  
3. الأعضاء التي تساهم فيه (Organs)
4. البُعد الزمني (Temporal Dimension)
5. الداخلي والخارجي (Internal vs External)

وما هي الأحكام التي تساعدنا على أن نحكم عليه بالسواء أو اللاسواء؟

وهل هذه الأمور تختلف حسب:
- الإنسان المؤمن
- المنافق
- الكافر

وأيضاً القلب الذي يمثل كل شخصية:
- القلب المريض يمثل المنافق
- القلب الميت يمثل الكافر
- وهكذا...

فما هي خارطة السلوك ضمن هذه المتغيرات التي ذكرناها؟

---

المطلوب (يجب إكمال جميع النقاط):
1. إجابة علمية موثقة بالآيات والمصادر
2. شرح المنهجية المتبعة في التحليل
3. جدول أو خارطة توضيحية للعلاقات
4. التفريق بين المؤمن والمنافق والكافر في كل محور
5. ربط القلب بالسلوك وأنماط الشخصية (القلب السليم، المريض، الميت، القاسي)

تنبيه مهم: يجب إكمال قسم أنماط القلوب بالكامل:
- القلب السليم (المؤمن) - بالتفصيل
- القلب المريض (المنافق) - بالتفصيل  
- القلب الميت (الكافر) - بالتفصيل
- القلب القاسي - بالتفصيل
- خارطة موحدة تربط جميع المتغيرات
"""

# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_academic_test():
    """Execute the academic research query and save results."""
    
    print("=" * 80)
    print("QBM ACADEMIC RESEARCH TEST v2 (Extended Token Limit)")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"API: {'Thesys C1' if BASE_URL else 'OpenAI'}")
    print(f"Model: {MODEL}")
    print(f"Max Tokens: {MAX_TOKENS}")
    print("\n" + "-" * 80)
    print("QUESTION:")
    print("-" * 80)
    print(ACADEMIC_QUESTION)
    print("\n" + "-" * 80)
    print("PROCESSING...")
    print("-" * 80 + "\n")
    
    # Initialize client
    if BASE_URL:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    else:
        client = OpenAI(api_key=API_KEY)
    
    model = MODEL
    
    try:
        # Make the API call with INCREASED TOKEN LIMIT
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACADEMIC_SCHOLAR_PROMPT},
                {"role": "user", "content": ACADEMIC_QUESTION}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=MAX_TOKENS,  # INCREASED FROM 8000 TO 16000
        )
        
        # Extract the response
        answer = response.choices[0].message.content
        
        print("=" * 80)
        print("SCHOLARLY RESPONSE")
        print("=" * 80)
        print(answer)
        print("\n" + "=" * 80)
        
        # Check for truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print("\n⚠️ WARNING: Response was truncated due to token limit!")
            print("   Consider increasing MAX_TOKENS or splitting the query.")
        else:
            print(f"\n✅ Response completed successfully (finish_reason: {finish_reason})")
        
        # Save to file
        output = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "max_tokens": MAX_TOKENS,
            "finish_reason": finish_reason,
            "question": ACADEMIC_QUESTION,
            "system_prompt_summary": "Academic Islamic Scholar with QBM Data (v2 - Extended)",
            "response": answer,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens if response.usage else None,
                "completion": response.usage.completion_tokens if response.usage else None,
                "total": response.usage.total_tokens if response.usage else None,
            }
        }
        
        # Save JSON
        output_json = Path(__file__).parent.parent.parent / "qbm_academic_test_result_v2.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # Save Markdown
        output_md = Path(__file__).parent.parent.parent / "qbm_academic_test_result_v2.md"
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(f"# نتائج البحث الأكاديمي - مشروع QBM (v2)\n\n")
            f.write(f"**التاريخ:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**النموذج المستخدم:** {model}\n\n")
            f.write(f"**الحد الأقصى للرموز:** {MAX_TOKENS}\n\n")
            f.write(f"**حالة الإكمال:** {finish_reason}\n\n")
            f.write("---\n\n")
            f.write("## السؤال البحثي\n\n")
            f.write(ACADEMIC_QUESTION)
            f.write("\n\n---\n\n")
            f.write("## الإجابة العلمية\n\n")
            f.write(answer)
            f.write("\n\n---\n\n")
            f.write("*تم إنشاء هذا التقرير آلياً بواسطة نظام QBM للبحث الأكاديمي (v2)*\n")
        
        print("\n✅ Results saved to:")
        print(f"   - {output_json}")
        print(f"   - {output_md}")
        
        return answer
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: Please set THESYS_API_KEY or OPENAI_API_KEY environment variable")
        print("\nExample:")
        print("  export THESYS_API_KEY=your_key_here")
        print("  python test_academic_query_v2.py")
        sys.exit(1)
    
    run_academic_test()
