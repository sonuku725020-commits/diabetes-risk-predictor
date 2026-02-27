# 🏥 Clinical Decision Support System for Diabetes Prevention

## Praxis 2.0 - Prototype Submission

---

## 📋 Overview

This is an enterprise-grade **Clinical Decision Support System (CDSS)** designed for **diabetes risk assessment and prevention**. The system leverages machine learning algorithms combined with evidence-based clinical guidelines to provide personalized risk predictions and intervention recommendations.

### Key Features

- **AI-Powered Risk Prediction**: Machine learning model with uncertainty quantification
- **Dual Interfaces**: Clinician dashboard and patient-friendly reports
- **Counterfactual Reasoning**: "What-if" scenario analysis for intervention planning
- **Longitudinal Tracking**: Patient history and trend analysis
- **Bias Detection**: Fairness metrics and demographic parity analysis
- **Evidence-Based Recommendations**: Based on major clinical trials (DPP, PREDIMED)
- **GenAI-Powered Insights**: AI-generated explanations, meal suggestions, and exercise plans using OpenRouter API

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐        ┌─────────────────────┐          │
│  │   Streamlit UI      │        │   FastAPI Backend   │          │
│  │   (Patient View)    │        │   (Clinician API)   │          │
│  └─────────────────────┘        └─────────────────────┘          │
├─────────────────────────────────────────────────────────────────────┤
│                      BUSINESS LOGIC LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              Clinical Decision Support Engine                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │ │
│  │  │ Risk          │  │ Intervention │  │ Counterfactual   │   │ │
│  │  │ Prediction    │  │ Generator    │  │ Analysis        │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │  ML Model       │  │ Evidence DB    │  │ Patient History │    │
│  │  (Pickle)       │  │ (Clinical)     │  │ (In-Memory)     │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Frontend | Streamlit | 1.29.0 |
| Backend | FastAPI | 0.104.1 |
| ML Framework | scikit-learn | 1.3.2 |
| Data Processing | Pandas, NumPy | 2.1.3 / 1.26.0 |
| Visualization | Plotly | 5.18.0 |
| Server | Uvicorn | 0.24.0 |
| GenAI | OpenRouter API (Claude) | Free Tier |

---

## 🔬 Approach & Methodology

### 1. Problem Statement

**Diabetes** is a chronic condition affecting **462 million people** worldwide. Early detection and preventive interventions can significantly reduce the risk of developing type 2 diabetes. However, many at-risk individuals remain unidentified until complications arise.

### 2. Solution Approach

Our Clinical Decision Support System addresses this by:

1. **Early Risk Identification**: Using ML to identify at-risk individuals before symptoms appear
2. **Uncertainty Quantification**: Providing confidence intervals to help clinicians understand prediction reliability
3. **Personalized Interventions**: Tailoring recommendations based on individual patient profiles
4. **Actionable Insights**: Translating complex medical data into understandable recommendations

### 3. ML Model Details

- **Algorithm**: Random Forest Classifier (Calibrated)
- **Features Used**:
  - Demographic: Age, Gender
  - Clinical: BMI, HbA1c, Blood Glucose, Blood Pressure
  - Medical History: Hypertension, Heart Disease, Smoking History
  - Lifestyle: Physical Activity, Diet Quality, Sleep, Stress

- **Risk Categories**:
  - Low: < 30%
  - Moderate: 30-60%
  - High: 60-85%
  - Critical: > 85%

### 4. Evidence Base

All recommendations are based on Level A evidence from:

- **Diabetes Prevention Program (DPP)**: 58% risk reduction with lifestyle intervention
- **PREDIMED Trial**: 30% risk reduction with Mediterranean diet
- **ADA Clinical Practice Guidelines**: American Diabetes Association standards
- **USPSTF Recommendations**: U.S. Preventive Services Task Force guidelines

---

## 📝 Assumptions

### Data Assumptions

1. The input data follows standard clinical measurement units (BMI in kg/m², HbA1c in %, glucose in mg/dL)
2. The model was trained on a representative dataset covering diverse demographics
3. Missing lifestyle data is handled gracefully with default moderate values

### System Assumptions

1. This is a prototype for demonstration purposes
2. In production, patient data would be stored in a HIPAA-compliant database
3. The ML model requires periodic recalibration (recommended: quarterly)
4. API authentication would be implemented for production use

### Clinical Assumptions

1. Recommendations supplement, not replace, clinical judgment
2. Users should consult healthcare providers before making medical decisions
3. The system is designed for adults (age 18+) only

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository** (if applicable)
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### GenAI Setup (Optional)

To enable AI-powered health insights:

1. Get a free API key from [OpenRouter.ai](https://openrouter.ai/)
2. Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```
3. Restart the application

### Running the Application

#### Option 1: Streamlit UI (Recommended for Demo)

```bash
streamlit run app.py
```

This opens the interactive web interface at `http://localhost:8501`

#### Option 2: FastAPI Backend

```bash
uvicorn main:app --reload
```

API documentation available at `http://localhost:8000/docs`

---

## 📊 Usage Guide

### For Patients

1. **Enter your health information** in the sidebar
2. **View your risk assessment** with detailed explanations
3. **Review personalized recommendations** tailored to your profile
4. **Track your progress** over time

### For Clinicians

1. **Access the API** at `/docs` endpoint
2. **Submit patient data** via POST requests
3. **Receive comprehensive reports** including:
   - Risk prediction with confidence intervals
   - Key risk factors analysis
   - Evidence-based intervention recommendations
   - Counterfactual scenarios
   - Follow-up scheduling

---

## 🔒 Safety & Ethics

### Clinical Safety

- ⚠️ This is a **prototype/demo** system
- ⚠️ Not for actual clinical diagnosis without proper validation
- ⚠️ All recommendations should be reviewed by healthcare professionals
- ⚠️ System includes safety alerts for critical values

### Bias & Fairness

The system includes:

- Demographic parity analysis
- Calibration by group analysis
- Bias detection and reporting
- Recommendations for mitigation

### Limitations

- Model accuracy depends on data quality
- Predictions may not generalize to all populations
- Lifestyle factors are self-reported
- Does not account for all possible comorbidities

---

## 📈 Business Feasibility

### Market Opportunity

| Metric | Value |
|--------|-------|
| Global Diabetes Market | $500B+ |
| Addressable Market | $50B (prevention) |
| Target Users | Healthcare providers, at-risk patients |

### Competitive Advantages

1. **Integrated Solution**: Combines prediction, tracking, and recommendations
2. **Dual Interface**: Serves both clinicians and patients
3. **Evidence-Based**: Built on proven clinical trials
4. **Explainable AI**: Uncertainty quantification and counterfactuals

### Implementation Costs (Estimated)

| Component | Development Effort |
|-----------|-------------------|
| ML Model | 2-4 weeks |
| Backend API | 3-4 weeks |
| Frontend UI | 4-6 weeks |
| Testing/Validation | 4-8 weeks |
| **Total** | **3-4 months** |

### Revenue Model Options

1. **B2B**: Healthcare systems, clinics (SaaS subscription)
2. **B2C**: Direct-to-consumer (freemium model)
3. **Licensing**: IP licensing to EHR vendors

---

## 📁 File Structure

```
.
├── app.py                      # Streamlit frontend application
├── main.py                     # FastAPI backend application  
├── diabetes_clinical_model_calibrated.pkl  # Trained ML model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔗 API Endpoints (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit patient data for risk assessment |
| `/report/clinician/{patient_id}` | GET | Get detailed clinician report |
| `/report/patient/{patient_id}` | GET | Get patient-friendly report |
| `/counterfactuals/{patient_id}` | GET | Get what-if scenario analysis |
| `/history/{patient_id}` | GET | Get patient history |
| `/bias-analysis` | POST | Analyze model fairness |

---

## 📞 Support & Documentation

- **Demo**: Run `streamlit run app.py` for interactive demo
- **API Docs**: Visit `/docs` when running FastAPI
- **Model Features**: See inline documentation in code

---

## 📄 License

This is a prototype for educational/demonstration purposes.

---

## 🙏 Acknowledgments

- Diabetes Prevention Program (DPP) Research Group
- PREDIMED Trial Investigators
- American Diabetes Association
- scikit-learn community
- Streamlit team
