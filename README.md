# Weather Condition Classification using Support Vector Machine (SVM) and Open-Meteo API

**AI-ML Assignment – 6**  
**Submission Deadline:** 28 July 2026, 11:59 PM IST  
**Student Details:**
- **Name:** Jai Arora
- **Registration Number:** 23BAI10546
- **Enrollment Number:** IN26011306

---

## 📌 Objective
The goal of this project is to build and evaluate a Support Vector Machine (SVM) classification model that predicts whether weather conditions are **Warm** ($\ge 25^\circ\text{C}$) or **Cool** ($< 25^\circ\text{C}$) based on hourly meteorological observations collected live from the **Open-Meteo Weather API**.

---

## 🌐 API Documentation Link
- **API Documentation:** [https://open-meteo.com/](https://open-meteo.com/)
- **Data Endpoint:** `https://api.open-meteo.com/v1/forecast` (Free access, no API key required)

---

## 🧰 Libraries Used
- **Data Collection & Manipulation:** `requests`, `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `LabelEncoder`, `train_test_split`, `SVC`, metrics)
- **Data Visualization:** `matplotlib`, `seaborn`
- **Development Environment:** Jupyter Notebook / Python 3.x

---

## 🔬 Methodology

### 1. Data Collection & Understanding (Task 1)
- Fetched 1,680 hourly weather observation records from the Open-Meteo API across multiple global locations (New Delhi, London, Tokyo, Cairo, Sydney).
- Extracted key meteorological variables into a Pandas DataFrame:
  - **Input Features ($X$):**
    1. `temperature_2m`: Air temperature at 2 meters above ground (°C)
    2. `relative_humidity_2m`: Relative humidity at 2 meters (%)
    3. `surface_pressure`: Surface air pressure (hPa)
    4. `wind_speed_10m`: Wind speed at 10 meters (km/h)
  - **Target Variable ($y$):** `Weather_Class`
    - **Warm** $\rightarrow$ `temperature_2m` $\ge 25^\circ\text{C}$
    - **Cool** $\rightarrow$ `temperature_2m` $< 25^\circ\text{C}$

### 2. Data Preprocessing (Task 2)
- Verified dataset integrity: **0 missing/NULL values**.
- Dropped non-predictive columns (`time`, `location`).
- Encoded target variable `Weather_Class` using `LabelEncoder` (`Cool` $\rightarrow$ 0, `Warm` $\rightarrow$ 1).
- Split dataset into **80% Training set (1,344 samples)** and **20% Testing set (336 samples)** using stratified splitting (`random_state=42`).
- Scaled features using **`StandardScaler`** to ensure zero mean ($\mu=0$) and unit variance ($\sigma=1$).

### 3. Model Development (Task 3)
- Constructed an SVM Classifier (`SVC`) with **Radial Basis Function (RBF Kernel)**.
- Hyperparameters: `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `random_state=42`.
- Fitted model on scaled training data and predicted classes on unseen test data.

---

## 📊 Results & Performance Evaluation (Task 4)

### Evaluation Metrics
| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.9881** | **98.81%** |
| **Precision** | **0.9947** | **99.47%** |
| **Recall** | **0.9841** | **98.41%** |
| **F1-Score** | **0.9894** | **98.94%** |

### Confusion Matrix
```
               Predicted Cool    Predicted Warm
Actual Cool         146                1
Actual Warm           3              186
```

### Key Observations
1. **Exceptional Accuracy:** The RBF-kernel SVM model achieved **98.81% accuracy**, accurately separating Warm and Cool weather states.
2. **High Precision & Recall:** Precision (**99.47%**) and Recall (**98.41%**) confirm minimal false positives and false negatives, reflecting a highly reliable decision boundary.
3. **Distinct Feature Separability:** Out of 336 test instances, only 4 were misclassified, proving that standardizing humidity, surface pressure, temperature, and wind speed yields strong linear/non-linear separability in kernel space.

---

## 📝 Conclusion (Task 5)

> This project successfully developed an SVM classification model using RBF kernel to predict weather conditions (Warm vs Cool) based on Open-Meteo API data. The model achieved an exceptional accuracy of over 98%, demonstrating high precision and recall. Feature scaling via StandardScaler was crucial for model performance, as SVM relies on distance calculations (Euclidean distance in kernel space); without standardization, features with larger magnitudes like surface pressure (~1013 hPa) would dominate temperature (~25°C) and wind speed. A key advantage of the SVM algorithm is its effectiveness in high-dimensional spaces and robust margin-maximization capability via non-linear RBF kernel mapping. However, a notable limitation is its high computational complexity and memory requirement on large-scale datasets, alongside sensitivity to hyperparameter selection.

---

## 📂 Repository Structure
```
├── Assignment-6.ipynb      # Complete Jupyter Notebook with execution outputs and plots
├── Assignment-6.py         # Standalone Python script for end-to-end execution
├── confusion_matrix.png    # Saved seaborn confusion matrix plot
└── README.md               # Project documentation & summary
```

---

## 🚀 How to Run

1. **Clone Repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Install Required Libraries:**
   ```bash
   pip install requests pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Run Python Script:**
   ```bash
   python Assignment-6.py
   ```

4. **Or Open Jupyter Notebook:**
   ```bash
   jupyter notebook Assignment-6.ipynb
   ```

---

## 📋 Google Form Submission Checklist
- [x] **Name:** Your Name
- [x] **Registration Number:** Your Registration Number
- [x] **Application Number:** Your Application Number
- [x] **Batch Number:** Your Batch Number
- [x] **Assignment Number:** Assignment -6
- [x] **Public GitHub Repository Link:** `<https://github.com/YourUsername/Repository-Name>`
- [x] **Email Address:** Your Email Address
- [x] **Google Form Link:** [https://forms.gle/fFL2CFooc5Vb2MXq8](https://forms.gle/fFL2CFooc5Vb2MXq8)
