import gradio as gr
import joblib
import pandas as pd

model_data = joblib.load("istifa_model.pkl")
model = model_data["model"]
threshold = model_data["threshold"]
expected_columns = model_data["columns"]

sorular = [
    ("Age", "Kaç yaşındasın?", "number", None),
    ("BusinessTravel", "İş seyahat sıklığın nedir? (Travel_Rarely, Travel_Frequently, Non-Travel)", "category", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]),
    ("Department", "Departmanın nedir? (Sales, Research & Development, Human Resources)", "category", ["Sales", "Research & Development", "Human Resources"]),
    ("DistanceFromHome", "Ev ile işyeri arası mesafe (km)?", "number", None),
    ("Education", "Eğitim seviyen? (1-5)", "number", (1, 5)),
    ("EnvironmentSatisfaction", "Çevre memnuniyetin? (1-4)", "number", (1, 4)),
    ("Gender", "Cinsiyetin? (Male, Female)", "category", ["Male", "Female"]),
    ("JobInvolvement", "İşe katılım düzeyin? (1-4)", "number", (1, 4)),
    ("JobLevel", "Pozisyon seviyen?", "number", None),
    ("JobRole", "Pozisyonun nedir? (Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources)", "category", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]),
    ("JobSatisfaction", "İş memnuniyetin? (1-4)", "number", (1, 4)),
    ("MaritalStatus", "Medeni durumun? (Single, Married, Divorced)", "category", ["Single", "Married", "Divorced"]),
    ("MonthlyIncome", "Aylık gelir (₺)?", "number", None),
    ("NumCompaniesWorked", "Kaç farklı şirkette çalıştın?", "number", None),
    ("PercentSalaryHike", "Son maaş artış yüzdesi?", "number", None),
    ("PerformanceRating", "Performans puanın? (1-4)", "number", (1, 4)),
    ("RelationshipSatisfaction", "İlişki memnuniyetin? (1-4)", "number", (1, 4)),
    ("StockOptionLevel", "Hisse opsiyon seviyen? (0-3)", "number", (0, 3)),
    ("TotalWorkingYears", "Toplam iş deneyimin (yıl)?", "number", None),
    ("TrainingTimesLastYear", "Geçen yıl kaç eğitim aldın?", "number", None),
    ("WorkLifeBalance", "İş-yaşam dengesi? (1-4)", "number", (1, 4)),
    ("YearsAtCompany", "Şirkette kaç yıldır çalışıyorsun?", "number", None),
    ("YearsInCurrentRole", "Mevcut roldeki yıl sayın?", "number", None),
    ("YearsSinceLastPromotion", "Son terfiden bu yana geçen yıl?", "number", None),
    ("YearsWithCurrManager", "Mevcut yöneticinle geçen yıl?", "number", None),
]

session_data = {}

def validate_input(key, val, dtype, valid_values):
    if dtype == "number":
        try:
            num = int(val)
        except:
            return False, "Lütfen geçerli bir sayı girin."
        if valid_values is None:
            return True, ""
        if isinstance(valid_values, tuple):
            if num < valid_values[0] or num > valid_values[1]:
                return False, f"Lütfen {valid_values[0]} ile {valid_values[1]} arasında bir sayı girin."
        return True, ""
    elif dtype == "category":
        if val not in valid_values:
            return False, f"Lütfen aşağıdaki seçeneklerden birini girin: {', '.join(valid_values)}"
        return True, ""
    return True, ""

def chatbot(message, history):
    if "index" not in session_data:
        session_data["index"] = 0
        session_data["cevaplar"] = {}
        session_data["restart_prompt"] = False
        return "Merhaba! 👋 İstifa tahmin sistemine hoş geldin.\n" + sorular[0][1]

    if session_data.get("restart_prompt", False):
        # Kullanıcıdan y/n cevabı bekleniyor
        if message.strip().lower() == "y":
            session_data.clear()
            session_data["index"] = 0
            session_data["cevaplar"] = {}
            session_data["restart_prompt"] = False
            return "Yeniden başlıyoruz.\n" + sorular[0][1]
        elif message.strip().lower() == "n":
            session_data.clear()
            return "Görüşürüz! İyi günler."
        else:
            return "Lütfen sadece 'y' veya 'n' ile cevap ver.\nBaşlamak ister misin? (y/n)"

    index = session_data["index"]
    cevaplar = session_data["cevaplar"]

    if index < len(sorular):
        key, _, dtype, valid_values = sorular[index]
        is_valid, error_msg = validate_input(key, message.strip(), dtype, valid_values)
        if not is_valid:
            return error_msg + f"\nTekrar deneyelim:\n{sorular[index][1]}"
        cevaplar[key] = message.strip()
        session_data["index"] += 1

    if session_data["index"] == len(sorular):
        input_df = pd.DataFrame([cevaplar])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        proba = model.predict_proba(input_df)[:, 1][0]
        pred = int(proba >= threshold)
        status = "İstifa eder" if pred == 1 else "İstifa etmez"

        session_data["restart_prompt"] = True
        return f"✅ Tahmin sonucu: {status}\n📊 Olasılık: {round(proba, 4)}\n\nBaştan başlamak ister misin? (y/n)"

    soru = sorular[session_data["index"]][1]
    return soru

gr.ChatInterface(chatbot, title="💼 İstifa Tahmin Chatbot'u").launch()
