# 👶 Smart Care - Infant Cry Analysis System | نظام تحليل بكاء الرضع


## 🚀 Project Overview
**Smart Care** is an advanced AI system that leverages Deep Learning to analyze and understand infant needs through their crying patterns. It helps parents distinguish between different states such as hunger, pain, discomfort, or sleepiness.

### 📊 Dataset
The project utilizes the **Donate-a-cry** dataset hosted on Kaggle:
- **Link:** [abduulahikhmaies/donateacry-corpus](https://www.kaggle.com/datasets/abduulahikhmaies/donateacry-corpus)
- **Content:** Thousands of classified audio samples (Hungry, Belly Pain, Discomfort, Tired, Burping).

---

## ✨ Key Features
- **Precise Audio Analysis:** Uses MFCCs (Mel-frequency cepstral coefficients) for high-fidelity feature extraction.
- **Advanced AI Model:** CNN-based architecture optimized with SMOTE for balanced data and improved accuracy.
- **Interactive GUI (Tkinter):**
    - **File Prediction:** Upload and analyze existing audio files.
    - **Real-Time Prediction:** Record and analyze live infant cries via microphone.
- **Performance Visualization:** Includes Confusion Matrix and accuracy/loss plots for model evaluation.

---

## 🛠 Tech Stack
- **Languages:** Python
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** Librosa, Sounddevice, Noisereduce
- **User Interface:** Tkinter
- **Data Science:** Scikit-learn, Pandas, NumPy, SMOTE
- **Visualization:** Matplotlib, Seaborn

---

## ⚙️ How It Works
1. **Preprocessing:** Audio cleaning and noise reduction.
2. **Feature Extraction:** Converting sounds into mathematical signatures (MFCC).
3. **Classification:** The model compares features against trained patterns.
4. **Insight:** Displays the reason for crying with a confidence score.

---

# 👶 Smart Care - نظام تحليل بكاء الرضع بالذكاء الاصطناعي

نظام ذكي متطور يستخدم تقنيات التعلم العميق (Deep Learning) لتحليل وفهم احتياجات الرضع من خلال نبرات بكائهم، مما يساعد الوالدين على فهم حالة طفلهم (جوع، ألم، نعاس، إلخ).

---

## 🚀 نظرة عامة على المشروع
يعتمد المشروع على معالجة الإشارات الصوتية واستخراج الخصائص الفريدة لكل نوع من أنواع البكاء باستخدام خوارزميات متقدمة، ثم تصنيفها باستخدام نموذج شبكة عصبية تلافيفية (CNN).

### 📊 البيانات المستخدمة
يستخدم المشروع مجموعة بيانات **Donate-a-cry** المتاحة على Kaggle:
- **الرابط:** [abduulahikhmaies/donateacry-corpus](https://www.kaggle.com/datasets/abduulahikhmaies/donateacry-corpus)
- **المحتوى:** آلاف العينات الصوتية المصنفة لأسباب بكاء مختلفة (Hungry, Belly Pain, Discomfort, Tired, Burping).

---

## ✨ المميزات الرئيسية
- **تحليلات دقيقة:** معالجة الصوت باستخدام MFCCs لاستخراج أدق التفاصيل الصوتية.
- **ذكاء اصطناعي متطور:** نموذج CNN مدرب ومعالج بتقنية SMOTE لموازنة البيانات وضمان دقة التصنيف.
- **واجهة مستخدم تفاعلية (GUI):** واجهة بسيطة تمكن المستخدم من:
    - رفع ملفات صوتية وتحليلها فوراً.
    - **التنبؤ اللحظي (Real-Time):** تسجيل صوت الطفل وتحليله مباشرة عبر الميكروفون.
- **تقارير أداء:** عرض مصفوفة الارتباك (Confusion Matrix) ومنحنيات الدقة لمتابعة جودة النموذج.

---

## 🛠 التكنولوجيا المستخدمة
- **البرمجة:** Python
- **التعلم العميق:** TensorFlow / Keras
- **معالجة الصوت:** Librosa, Sounddevice, Noisereduce
- **واجهة المستخدم:** Tkinter
- **معالجة البيانات:** Scikit-learn, Pandas, NumPy, SMOTE
- **الرسوم البيانية:** Matplotlib, Seaborn

---

## ⚙️ كيف يعمل النظام؟
1. **تجهيز البيانات:** يتم تنظيف الصوت وتقليل الضوضاء.
2. **استخراج الخصائص:** تحويل الموجات الصوتية إلى تمثيلات رياضية (MFCC).
3. **التصنيف:** يقوم النموذج بمقارنة الخصائص مع البيانات التي تدرب عليها.
4. **النتيجة:** يظهر للمستخدم سبب البكاء مع نسبة الثقة في التوقع.

---
تم تطوير هذا المشروع كحل ذكي لدعم الرعاية الصحة المنزلية. 🩺✨
