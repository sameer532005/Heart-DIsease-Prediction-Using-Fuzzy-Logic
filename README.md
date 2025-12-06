
# Heart Disease Risk Prediction using Fuzzy Logic 🫀

A web-based decision support system that predicts **heart disease risk using fuzzy logic**.  
Built with **Flask (Python)** + **HTML/CSS** + **SQLite**, featuring a **modern medical dashboard UI** with patient storage and history lookup.

---

## 📌 Features

| Module | Description |
|-------|-------------|
| 🔐 Login System | Secured entry for dashboard access |
| ➕ Add Patient | Fill clinical input fields and generate prediction |
| 🧠 Fuzzy Logic Model | Triangular MFs + rule-based inference + weighted defuzzification |
| 📊 Dashboard Visualization | Displays risk score, risk level, membership activations & summary |
| 🗄️ SQLite Storage | Saves every prediction with timestamp + patient name |
| 📁 Patient History | View all previously tested patients and reopen results |
| UI Inspired Dashboard | Clean heart-themed interface with modern styling |

---

## 🧪 Fuzzy Logic Model

The system applies fuzzy membership to heart-related clinical parameters using triangular membership functions.  
Rules decide activation levels for **Low / Medium / High** risk.  
Defuzzification converts fuzzy output to a numerical score:

```
Risk Score = (Low*20 + Medium*50 + High*80) / (Low+Medium+High)
```

Final result is displayed as:

| Output | Description |
|--------|-------------|
| **Risk Score (%)** | Defuzzified percentage value |
| **Risk Category** | Low / Medium / High |
| **Membership Activation** | Fuzzy strength of L/M/H contribution |

---

## 📂 Folder Structure

```
project/
│── app.py                  # Flask backend + fuzzy model + DB integration
│── predictions.db          # Auto-generated SQLite database
│── templates/              # Frontend HTML pages
│   ├── index.html
│   ├── dashboard.html
│   ├── input.html
│   ├── result.html
│   └── patients.html
│── static/
│   ├── style.css
│   └── heart.png
│── README.md
```

---

## 🛠 Installation & Setup

### 1️⃣ Clone Repo

```
git clone <your_repo_url>
cd project_folder
```

### 2️⃣ Install Dependencies

`sqlite3` already included in Python. Install others:

```
pip install flask pandas numpy
```

### 3️⃣ Update dataset path in `app.py`:

```
DATA_PATH = "your_dataset_path.csv"
```

### 4️⃣ Run App

```
python app.py
```

### 5️⃣ Open Browser

```
http://127.0.0.1:5000
```

➡ Default Login:

```
Username: admin
Password: 1234
```

---

## 🚀 How Application Works

1. Login to access dashboard  
2. Choose **Add Patient & Predict**  
3. Fill patient clinical details → Submit  
4. Dashboard shows prediction with:
   - Risk score
   - Fuzzy activations
   - Summary table
   - Input parameter table  
5. Prediction automatically saved to database  
6. History view available under **View All Patients**  
7. Click any patient record to reopen full result dashboard  

---

## 📊 Example Output (Dashboard)

*(Insert screenshots later)*

---

## 📝 Future Improvements

- Hybrid ML + Fuzzy Logic model
- Live sensor integration
- Doctor/patient multi-user login
- Export report as PDF

---

## 👥 Team Members

| Name | Role |
|-----|------|
| Member 1 | Backend & Fuzzy Logic |
| Member 2 | UI/Frontend |
| Member 3 | Database & Integration |

---

Give the repository a ⭐ if you found it useful!
