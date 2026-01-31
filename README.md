# 📉 EmployeeLoss (Employee Retention AI Bot)

![Python](https://img.shields.io/badge/python-3.x-blue?style=flat&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=flat)
![Interface](https://img.shields.io/badge/Interface-Gradio-yellow?style=flat)
![Status](https://img.shields.io/badge/Status-Model%20Showcase-lightgrey?style=flat)

**EmployeeLoss** is an AI-powered chatbot designed to analyze employee feedback, detect sentiment, and categorize complaints/suggestions to improve company culture and reduce turnover.

This repository serves as a **showcase for the serialized model (`.pkl`) and architecture**. Due to the presence of proprietary company data and Personally Identifiable Information (PII), the full source code and dataset are not publicly available.

---

## 🤖 System Overview & Architecture

The system operates as an internal intelligence tool for HR departments. It processes raw data collected from employees and provides actionable insights through a conversational interface.

### 🔄 The Workflow
1.  **Data Collection:** Employees submit complaints, comments, and suggestions via **Google Forms**.
2.  **Preprocessing:** Raw text data is cleaned and structured using **Pandas**.
3.  **NLP Analysis:** The core model (built with **Scikit-learn**) performs:
    * **Sentiment Analysis:** Determining the emotional tone of the feedback.
    * **Categorization:** Classifying feedback into specific topics (e.g., Salary, Management, Cafeteria, Work Hours).
4.  **Interaction:** A **Gradio** web interface allows authorized users to query the data naturally.

---

## 🔐 Security & Access Control

* **Domain-Restricted Auth:** The chatbot includes an authentication layer that only allows login via specific company email domains, ensuring sensitive employee feedback remains confidential.
* **Privacy First:** The public repository does not contain the training data. Only the trained model structure (`.pkl`) is archived here.

---

## 🧠 Technical Stack

* **Python 3:** Core programming language.
* **Scikit-learn:** Used for building the classification and sentiment analysis models.
* **Pandas:** Used for data manipulation and creating the dataframe from Google Forms exports.
* **Joblib:** Used for model serialization and persistence (saving/loading the `.pkl` file).
* **Gradio:** Used to create the interactive web interface for the chatbot.

---

## 💬 Capabilities & Example Queries

The model is trained on **Turkish** language datasets and supports queries such as:

* *"How many suggestions do we have regarding the cafeteria?"* (Kafeterya hakkında kaç öneri var?)
* *"What is the most frequent complaint topic this month?"* (Bu ay en çok hangi konuda şikayet var?)
* *"Show me the negative feedback regarding overtime."* (Mesai saatleri ile ilgili olumsuz geri bildirimleri göster.)

---

## ⚠️ Note on Usage

This repository contains the `model.pkl` file. However, without the proprietary server environment and the specific vectorizer vocabulary used during training, the model cannot be deployed locally by third parties. It is hosted here for **portfolio demonstration purposes only**.

---

## 👨‍💻 Author

**Ozan Arda Özçelik**

- **GitHub:** [@ozanardaozcelik](https://github.com/ozanardaozcelik)
- **LinkedIn:** [Ozan Arda Özçelik](https://www.linkedin.com/in/ozanardaozcelik/)
