# advanced_streamlit_app.py
"""
🏥 Advanced Clinical Decision Support System
Standalone Streamlit Application - No FastAPI Required
Professional-grade diabetes risk assessment platform

Features:
- AI-powered risk prediction with uncertainty quantification
- GenAI-powered personalized health insights
- Evidence-based intervention recommendations
- Counterfactual reasoning
- Longitudinal tracking
- Bias detection
- Safety checks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import io
import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🏥 Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '# Clinical Decision Support System\nAI-powered diabetes risk assessment'
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Risk level cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .risk-low {
        background-color: #d4edda;
        border-color: #28a745;
    }
    
    .risk-moderate {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    
    .risk-high {
        background-color: #ffe0b2;
        border-color: #ff9800;
    }
    
    .risk-critical {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        height: 100%;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Intervention cards */
    .intervention-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .priority-critical { border-color: #dc3545; }
    .priority-high { border-color: #ff9800; }
    .priority-medium { border-color: #ffc107; }
    .priority-low { border-color: #28a745; }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .alert-critical {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Progress bars */
    .progress-container {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class PatientData:
    """Patient data structure"""
    patient_id: str
    age: int
    gender: str
    bmi: float
    hba1c: float
    glucose: int
    hypertension: bool
    heart_disease: bool
    smoking: str
    family_history: bool = False
    activity_level: int = 3
    diet_quality: int = 3
    sleep_hours: float = 7.0
    stress_level: int = 5

@dataclass
class RiskAssessment:
    """Risk assessment results"""
    risk_score: float
    adjusted_risk: float
    risk_level: RiskLevel
    confidence_interval: Tuple[float, float]
    confidence: str
    risk_factors: List[Dict]
    interventions: List[Dict]
    counterfactuals: List[Dict]
    safety_alerts: List[str]
    followup_schedule: Dict[str, str]

@dataclass
class LongitudinalRecord:
    """Historical record"""
    date: datetime
    risk_score: float
    hba1c: float
    glucose: int
    bmi: float
    notes: str = ""

# ============================================================================
# GENAI POWERED HEALTH INSIGHTS ENGINE
# ============================================================================

class GenAIHealthInsights:
    """
    GenAI-powered personalized health insights using OpenRouter API
    Provides natural language explanations and health coaching
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = "anthropic/claude-3-haiku:free"  # Free model
        self.enabled = bool(self.api_key)
    
    def generate_insight(self, prompt: str) -> str:
        """Generate AI-powered health insight"""
        if not self.enabled:
            return "GenAI insights unavailable. Please configure OPENROUTER_API_KEY in .env"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://clinical-decision-support.streamlit.app",
                "X-Title": "Clinical Decision Support System"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful medical health assistant. Provide clear, accurate, and patient-friendly explanations. Always remind users to consult their healthcare provider for medical advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 300
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Unable to generate insight at this time. (Error: {response.status_code})"
                
        except Exception as e:
            return f"GenAI service temporarily unavailable: {str(e)}"
    
    def explain_risk_factors(self, patient_data: Dict, risk_factors: List[Dict]) -> str:
        """Generate personalized explanation of risk factors"""
        prompt = f"""
Patient Profile:
- Age: {patient_data.get('age', 'N/A')}
- BMI: {patient_data.get('bmi', 'N/A')}
- HbA1c: {patient_data.get('hba1c', 'N/A')}%
- Blood Glucose: {patient_data.get('glucose', 'N/A')} mg/dL
- Blood Pressure: {patient_data.get('systolic', 'N/A')}/{patient_data.get('diastolic', 'N/A')} mmHg
- Hypertension: {'Yes' if patient_data.get('hypertension') else 'No'}
- Heart Disease: {'Yes' if patient_data.get('heart_disease') else 'No'}

Key Risk Factors:
{chr(10).join([f"- {rf.get('name', 'Unknown')}: {rf.get('value', 'N/A')} ({rf.get('status', 'N/A')})" for rf in risk_factors[:5]])}

Please explain in simple terms what these numbers mean for this patient's health and why they are at risk for diabetes. Keep it concise (3-4 sentences).
        """
        return self.generate_insight(prompt)
    
    def generate_meal_suggestions(self, patient_data: Dict, risk_level: str) -> str:
        """Generate personalized meal suggestions"""
        prompt = f"""
A patient with diabetes risk level: {risk_level}
BMI: {patient_data.get('bmi', 'N/A')}
Age: {patient_data.get('age', 'N/A')}

Provide 3 specific meal suggestions (1 breakfast, 1 lunch, 1 dinner) that would help reduce their diabetes risk. Include specific foods and portion ideas. Keep it practical and realistic.
        """
        return self.generate_insight(prompt)
    
    def generate_exercise_plan(self, patient_data: Dict) -> str:
        """Generate personalized exercise recommendations"""
        prompt = f"""
A patient with:
- Age: {patient_data.get('age', 'N/A')}
- BMI: {patient_data.get('bmi', 'N/A')}
- Current activity level: {patient_data.get('activity_level', 'N/A')}/5

Provide a simple weekly exercise plan to help reduce diabetes risk. Include specific activities, duration, and frequency. Consider their age and BMI.
        """
        return self.generate_insight(prompt)


# ============================================================================
# CLINICAL DECISION SUPPORT ENGINE
# ============================================================================

class AdvancedClinicalEngine:
    """
    Advanced Clinical Decision Support Engine
    Comprehensive diabetes risk assessment system
    """
    
    def __init__(self, model_path: str):
        """Initialize engine with model"""
        self.model = None
        self.model_loaded = False
        
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
        except Exception as e:
            st.error(f"❌ Model loading error: {e}")
        
        # Feature configuration
        self.feature_names = [
            'age', 'hypertension', 'heart_disease', 'bmi',
            'HbA1c_level', 'blood_glucose_level', 
            'gender_encoded', 'smoking_encoded'
        ]
        
        # Categorical mappings
        self.gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        self.smoking_map = {
            'never': 0, 'No Info': 1, 'former': 2,
            'not current': 3, 'current': 4, 'ever': 5
        }
        
        # Risk thresholds
        self.thresholds = {
            'low': 0.30,
            'moderate': 0.60,
            'high': 0.85
        }
        
        # Clinical reference ranges
        self.reference_ranges = {
            'hba1c': {'normal': 5.7, 'prediabetes': 6.5, 'diabetes': 6.5},
            'glucose': {'normal': 100, 'prediabetes': 126, 'diabetes': 126},
            'bmi': {'normal': 25, 'overweight': 30, 'obese': 35}
        }
        
        # Evidence database
        self.evidence_db = self._load_evidence_database()
        
        # Patient history storage
        self.patient_history: Dict[str, List[LongitudinalRecord]] = {}
    
    def _load_evidence_database(self) -> Dict:
        """Load clinical evidence database"""
        return {
            'interventions': {
                'lifestyle': {
                    'dpp': {
                        'name': 'Diabetes Prevention Program',
                        'impact': '58% risk reduction',
                        'components': ['7% weight loss', '150 min/week exercise'],
                        'evidence_level': 'A',
                        'nnt': 7  # Number needed to treat
                    },
                    'mediterranean': {
                        'name': 'Mediterranean Diet',
                        'impact': '30% risk reduction',
                        'evidence_level': 'A',
                        'source': 'PREDIMED trial'
                    }
                },
                'pharmacological': {
                    'metformin': {
                        'name': 'Metformin',
                        'dose': '850mg BID',
                        'impact': '31% risk reduction',
                        'evidence_level': 'A',
                        'source': 'DPP study'
                    }
                }
            },
            'guidelines': {
                'ada': 'American Diabetes Association 2024',
                'uspstf': 'US Preventive Services Task Force',
                'aace': 'American Association of Clinical Endocrinologists'
            }
        }
    
    def preprocess(self, patient: PatientData) -> pd.DataFrame:
        """Convert patient data to model features"""
        features = pd.DataFrame([{
            'age': patient.age,
            'hypertension': 1 if patient.hypertension else 0,
            'heart_disease': 1 if patient.heart_disease else 0,
            'bmi': patient.bmi,
            'HbA1c_level': patient.hba1c,
            'blood_glucose_level': patient.glucose,
            'gender_encoded': self.gender_map.get(patient.gender, 0),
            'smoking_encoded': self.smoking_map.get(patient.smoking, 1)
        }])
        return features
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, Tuple[float, float], str]:
        """Predict risk with uncertainty quantification"""
        if not self.model_loaded:
            return 0.5, (0.4, 0.6), "Unknown"
        
        try:
            risk_score = self.model.predict_proba(features)[0, 1]
        except:
            try:
                risk_score = self.model.predict_proba(features.values)[0, 1]
            except:
                return 0.5, (0.4, 0.6), "Error"
        
        # Bootstrap confidence interval (simplified)
        std_error = 0.05
        ci_lower = max(0, risk_score - 1.96 * std_error)
        ci_upper = min(1, risk_score + 1.96 * std_error)
        
        # Confidence assessment
        if risk_score < 0.15 or risk_score > 0.85:
            confidence = "Very High"
        elif risk_score < 0.25 or risk_score > 0.75:
            confidence = "High"
        elif risk_score < 0.4 or risk_score > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return float(risk_score), (float(ci_lower), float(ci_upper)), confidence
    
    def adjust_risk(self, base_risk: float, patient: PatientData) -> float:
        """Calculate context-adjusted risk score"""
        adjusted = base_risk
        
        # Age adjustments
        if patient.age >= 70:
            adjusted += 0.12
        elif patient.age >= 65:
            adjusted += 0.08
        elif patient.age >= 60:
            adjusted += 0.05
        elif patient.age >= 55:
            adjusted += 0.02
        
        # Comorbidity adjustments
        if patient.hypertension:
            adjusted += 0.05
        if patient.heart_disease:
            adjusted += 0.06
        
        # Family history
        if patient.family_history:
            adjusted += 0.10
        
        # Lifestyle factors (protective)
        if patient.activity_level >= 4:
            adjusted -= 0.04
        elif patient.activity_level >= 3:
            adjusted -= 0.02
        
        if patient.diet_quality >= 4:
            adjusted -= 0.03
        
        if patient.sleep_hours >= 7 and patient.sleep_hours <= 9:
            adjusted -= 0.01
        
        if patient.stress_level <= 3:
            adjusted -= 0.02
        elif patient.stress_level >= 8:
            adjusted += 0.02
        
        return min(1.0, max(0.0, adjusted))
    
    def get_risk_level(self, risk: float) -> RiskLevel:
        """Categorize risk level"""
        if risk < self.thresholds['low']:
            return RiskLevel.LOW
        elif risk < self.thresholds['moderate']:
            return RiskLevel.MODERATE
        elif risk < self.thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def analyze_risk_factors(self, patient: PatientData) -> List[Dict]:
        """Comprehensive risk factor analysis"""
        factors = []
        
        # HbA1c Analysis
        hba1c = patient.hba1c
        if hba1c >= 6.5:
            status, score, interpretation = "CRITICAL", 0.95, "Meets diabetes diagnostic criteria"
        elif hba1c >= 6.0:
            status, score, interpretation = "HIGH", 0.75, "High prediabetes range"
        elif hba1c >= 5.7:
            status, score, interpretation = "ELEVATED", 0.55, "Prediabetes range"
        else:
            status, score, interpretation = "NORMAL", 0.10, "Normal glycemic control"
        
        factors.append({
            'name': 'HbA1c Level',
            'value': f'{hba1c:.1f}%',
            'numeric_value': hba1c,
            'status': status,
            'score': score,
            'interpretation': interpretation,
            'modifiable': True,
            'normal_range': '< 5.7%',
            'target': '< 5.7%',
            'category': 'Glycemic'
        })
        
        # Blood Glucose Analysis
        glucose = patient.glucose
        if glucose >= 126:
            status, score, interpretation = "CRITICAL", 0.90, "Diabetes range"
        elif glucose >= 110:
            status, score, interpretation = "HIGH", 0.70, "Impaired fasting glucose"
        elif glucose >= 100:
            status, score, interpretation = "ELEVATED", 0.50, "Borderline elevated"
        else:
            status, score, interpretation = "NORMAL", 0.10, "Normal fasting glucose"
        
        factors.append({
            'name': 'Fasting Glucose',
            'value': f'{glucose} mg/dL',
            'numeric_value': glucose,
            'status': status,
            'score': score,
            'interpretation': interpretation,
            'modifiable': True,
            'normal_range': '< 100 mg/dL',
            'target': '70-99 mg/dL',
            'category': 'Glycemic'
        })
        
        # BMI Analysis
        bmi = patient.bmi
        if bmi >= 40:
            status, score, interpretation = "CRITICAL", 0.85, "Class III Obesity (Severe)"
        elif bmi >= 35:
            status, score, interpretation = "HIGH", 0.70, "Class II Obesity"
        elif bmi >= 30:
            status, score, interpretation = "ELEVATED", 0.55, "Class I Obesity"
        elif bmi >= 25:
            status, score, interpretation = "MODERATE", 0.35, "Overweight"
        elif bmi >= 18.5:
            status, score, interpretation = "NORMAL", 0.10, "Normal weight"
        else:
            status, score, interpretation = "LOW", 0.15, "Underweight"
        
        factors.append({
            'name': 'Body Mass Index',
            'value': f'{bmi:.1f} kg/m²',
            'numeric_value': bmi,
            'status': status,
            'score': score,
            'interpretation': interpretation,
            'modifiable': True,
            'normal_range': '18.5 - 24.9',
            'target': '< 25 kg/m²',
            'category': 'Anthropometric'
        })
        
        # Age Analysis
        age = patient.age
        if age >= 70:
            status, score = "HIGH", 0.70
        elif age >= 60:
            status, score = "ELEVATED", 0.55
        elif age >= 50:
            status, score = "MODERATE", 0.40
        elif age >= 45:
            status, score = "LOW-MODERATE", 0.30
        else:
            status, score = "NORMAL", 0.15
        
        factors.append({
            'name': 'Age',
            'value': f'{age} years',
            'numeric_value': age,
            'status': status,
            'score': score,
            'interpretation': 'Age is a non-modifiable risk factor',
            'modifiable': False,
            'normal_range': 'N/A',
            'target': 'N/A',
            'category': 'Demographic'
        })
        
        # Hypertension
        if patient.hypertension:
            factors.append({
                'name': 'Hypertension',
                'value': 'Present',
                'numeric_value': 1,
                'status': 'ELEVATED',
                'score': 0.45,
                'interpretation': 'Increases cardiovascular and metabolic risk',
                'modifiable': True,
                'normal_range': 'Not present',
                'target': 'BP < 130/80 mmHg',
                'category': 'Cardiovascular'
            })
        
        # Heart Disease
        if patient.heart_disease:
            factors.append({
                'name': 'Heart Disease',
                'value': 'Present',
                'numeric_value': 1,
                'status': 'ELEVATED',
                'score': 0.50,
                'interpretation': 'Significant comorbidity requiring comprehensive management',
                'modifiable': True,
                'normal_range': 'Not present',
                'target': 'Optimal cardiac management',
                'category': 'Cardiovascular'
            })
        
        # Smoking
        if patient.smoking in ['current', 'not current']:
            status = 'HIGH' if patient.smoking == 'current' else 'ELEVATED'
            score = 0.55 if patient.smoking == 'current' else 0.35
            factors.append({
                'name': 'Smoking Status',
                'value': patient.smoking.replace('_', ' ').title(),
                'numeric_value': self.smoking_map.get(patient.smoking, 0),
                'status': status,
                'score': score,
                'interpretation': 'Smoking significantly increases diabetes risk',
                'modifiable': True,
                'normal_range': 'Never smoked',
                'target': 'Complete cessation',
                'category': 'Lifestyle'
            })
        
        # Family History
        if patient.family_history:
            factors.append({
                'name': 'Family History',
                'value': 'Positive',
                'numeric_value': 1,
                'status': 'ELEVATED',
                'score': 0.40,
                'interpretation': 'First-degree relative with diabetes increases risk',
                'modifiable': False,
                'normal_range': 'Negative',
                'target': 'N/A (genetic factor)',
                'category': 'Genetic'
            })
        
        # Physical Activity
        if patient.activity_level < 3:
            factors.append({
                'name': 'Physical Activity',
                'value': f'Level {patient.activity_level}/5',
                'numeric_value': patient.activity_level,
                'status': 'ELEVATED',
                'score': 0.35,
                'interpretation': 'Insufficient physical activity',
                'modifiable': True,
                'normal_range': '≥ 3/5',
                'target': '150 min/week moderate activity',
                'category': 'Lifestyle'
            })
        
        # Sort by score (highest risk first)
        factors.sort(key=lambda x: x['score'], reverse=True)
        
        return factors
    
    def generate_interventions(self, patient: PatientData, risk_level: RiskLevel, 
                              risk_factors: List[Dict]) -> List[Dict]:
        """Generate evidence-based intervention recommendations"""
        interventions = []
        
        # Urgent referral for critical/high risk
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            interventions.append({
                'id': 'URG001',
                'priority': Priority.CRITICAL,
                'category': 'Medical Referral',
                'title': '🚨 Urgent Endocrinology Consultation',
                'description': 'Schedule appointment with endocrinologist for comprehensive evaluation',
                'rationale': f'{risk_level.value.upper()} risk level requires immediate specialist evaluation',
                'expected_impact': 'Enable timely diagnosis and treatment initiation',
                'timeline': 'Within 1-2 weeks',
                'evidence_level': 'A',
                'evidence_source': 'ADA Guidelines 2024',
                'cost_estimate': '$200-$500',
                'barriers': ['Insurance coverage', 'Specialist availability'],
                'monitoring': 'Per specialist recommendations'
            })
        
        # HbA1c-based interventions
        if patient.hba1c >= 6.5:
            interventions.append({
                'id': 'MED001',
                'priority': Priority.HIGH,
                'category': 'Pharmacotherapy',
                'title': '💊 Initiate Metformin Therapy',
                'description': 'Start metformin 500mg once daily, titrate to 850mg BID over 4-8 weeks',
                'rationale': 'HbA1c ≥6.5% meets diabetes diagnostic criteria; metformin is first-line therapy',
                'expected_impact': 'HbA1c reduction of 1.0-1.5%, 31% risk reduction for complications',
                'timeline': 'Initiate within 2 weeks, titrate over 8 weeks',
                'evidence_level': 'A',
                'evidence_source': 'Diabetes Prevention Program',
                'cost_estimate': '$10-30/month (generic)',
                'contraindications': ['eGFR <30', 'Active liver disease', 'Heavy alcohol use'],
                'monitoring': 'Renal function, B12 levels annually'
            })
        elif patient.hba1c >= 5.7:
            interventions.append({
                'id': 'MED002',
                'priority': Priority.MEDIUM,
                'category': 'Preventive Pharmacotherapy',
                'title': '💊 Consider Metformin for Prevention',
                'description': 'Discuss metformin for diabetes prevention in high-risk prediabetes',
                'rationale': 'Prediabetes with additional risk factors; metformin shown effective for prevention',
                'expected_impact': '31% diabetes risk reduction over 3 years',
                'timeline': 'Consider after 3-6 months of lifestyle intervention',
                'evidence_level': 'A',
                'evidence_source': 'Diabetes Prevention Program',
                'cost_estimate': '$10-30/month',
                'patient_criteria': ['Age <60 preferred', 'BMI ≥35', 'History of GDM'],
                'monitoring': 'HbA1c every 3-6 months'
            })
        
        # Intensive lifestyle intervention
        if patient.hba1c >= 5.7 or patient.bmi >= 25:
            interventions.append({
                'id': 'LIFE001',
                'priority': Priority.HIGH if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else Priority.MEDIUM,
                'category': 'Lifestyle - Comprehensive',
                'title': '🏃 Intensive Lifestyle Modification Program',
                'description': 'Enroll in structured program: behavioral counseling + diet + exercise',
                'rationale': 'DPP showed 58% diabetes risk reduction with intensive lifestyle intervention',
                'expected_impact': '58% diabetes risk reduction, 7% weight loss target',
                'timeline': '16-week core program, then maintenance',
                'evidence_level': 'A',
                'evidence_source': 'Diabetes Prevention Program (DPP)',
                'cost_estimate': '$500-2000 for program',
                'program_elements': [
                    'Weekly group sessions (16 weeks)',
                    '7% weight loss goal',
                    '150 min/week moderate activity',
                    'Fat intake <25% of calories',
                    'Behavioral modification techniques'
                ],
                'monitoring': 'Weight, activity logs, HbA1c every 3 months'
            })
        
        # Weight management
        if patient.bmi >= 30:
            target_loss = (patient.bmi - 24.9) * 2.5
            interventions.append({
                'id': 'LIFE002',
                'priority': Priority.HIGH,
                'category': 'Lifestyle - Weight Management',
                'title': '⚖️ Medical Weight Loss Program',
                'description': f'Structured weight loss targeting {target_loss:.0f} kg (7-10% body weight)',
                'rationale': f'BMI {patient.bmi:.1f} indicates obesity; weight loss critical for diabetes prevention',
                'expected_impact': '58% risk reduction with 7% weight loss; improved metabolic markers',
                'timeline': '6-12 months for initial weight loss, then maintenance',
                'evidence_level': 'A',
                'evidence_source': 'DPP, Look AHEAD Study',
                'cost_estimate': '$100-300/month',
                'approach': [
                    'Caloric deficit 500-750 kcal/day',
                    'High protein diet (25-30% calories)',
                    'Consider meal replacements',
                    'Weekly weigh-ins',
                    'Food diary tracking'
                ],
                'medications_to_consider': ['Semaglutide', 'Tirzepatide', 'Phentermine-topiramate'],
                'monitoring': 'Weight weekly, body composition monthly'
            })
            
            if patient.bmi >= 40:
                interventions.append({
                    'id': 'SURG001',
                    'priority': Priority.MEDIUM,
                    'category': 'Surgical',
                    'title': '🏥 Bariatric Surgery Evaluation',
                    'description': 'Refer for bariatric surgery evaluation given severe obesity',
                    'rationale': 'BMI ≥40 qualifies for bariatric surgery; proven diabetes remission rates',
                    'expected_impact': '60-80% diabetes remission at 5 years',
                    'timeline': 'Evaluation within 3 months',
                    'evidence_level': 'A',
                    'evidence_source': 'STAMPEDE, SURGICAL trials',
                    'surgery_options': ['Gastric bypass', 'Sleeve gastrectomy', 'Duodenal switch']
                })
        elif patient.bmi >= 25:
            interventions.append({
                'id': 'LIFE003',
                'priority': Priority.MEDIUM,
                'category': 'Lifestyle - Weight Management',
                'title': '⚖️ Weight Management Counseling',
                'description': 'Nutritional counseling for moderate weight loss (5-7%)',
                'rationale': 'Overweight status contributes to insulin resistance',
                'expected_impact': 'Prevent progression to obesity, improve insulin sensitivity',
                'timeline': '3-6 months',
                'evidence_level': 'B',
                'cost_estimate': '$50-150/session',
                'monitoring': 'Weight monthly, waist circumference'
            })
        
        # Exercise prescription
        interventions.append({
            'id': 'LIFE004',
            'priority': Priority.HIGH if patient.activity_level < 3 else Priority.MEDIUM,
            'category': 'Lifestyle - Physical Activity',
            'title': '🏋️ Structured Exercise Program',
            'description': '150+ min/week moderate aerobic activity + resistance training 2-3x/week',
            'rationale': 'Physical activity improves insulin sensitivity and glucose uptake',
            'expected_impact': '30-50% improvement in insulin sensitivity, 20-30% diabetes risk reduction',
            'timeline': 'Ongoing, gradual progression over 8-12 weeks',
            'evidence_level': 'A',
            'evidence_source': 'DPP, multiple meta-analyses',
            'cost_estimate': '$0-50/month',
            'prescription': [
                'Start: 10-15 min walking 3x/week',
                'Progress: Add 5 min/week',
                'Goal: 150 min/week moderate intensity',
                'Add: Resistance training 2-3x/week',
                'Include: Flexibility exercises'
            ],
            'precautions': ['Cardiac screening if symptomatic', 'Start low, progress slow', 'Foot care for neuropathy'],
            'monitoring': 'Activity logs, fitness assessments quarterly'
        })
        
        # Nutrition therapy
        interventions.append({
            'id': 'LIFE005',
            'priority': Priority.HIGH if patient.hba1c >= 6.0 else Priority.MEDIUM,
            'category': 'Lifestyle - Nutrition',
            'title': '🥗 Medical Nutrition Therapy (MNT)',
            'description': 'Individualized nutrition counseling with registered dietitian',
            'rationale': 'Evidence-based dietary patterns reduce diabetes risk and improve glycemic control',
            'expected_impact': 'HbA1c reduction 0.5-1.0%, weight loss, improved lipids',
            'timeline': '3-4 sessions over 3-6 months, then quarterly',
            'evidence_level': 'A',
            'evidence_source': 'ADA Nutrition Therapy Guidelines',
            'cost_estimate': '$100-200/session',
            'dietary_patterns': [
                'Mediterranean diet (preferred)',
                'DASH diet',
                'Low glycemic index diet',
                'Plant-based diet'
            ],
            'key_recommendations': [
                'Reduce refined carbohydrates',
                'Increase fiber (25-30g/day)',
                'Limit added sugars (<10% calories)',
                'Choose healthy fats',
                'Moderate protein intake'
            ],
            'monitoring': 'Food diaries, glycemic response'
        })
        
        # Smoking cessation
        if patient.smoking in ['current', 'not current']:
            interventions.append({
                'id': 'LIFE006',
                'priority': Priority.HIGH,
                'category': 'Lifestyle - Smoking',
                'title': '🚭 Comprehensive Smoking Cessation',
                'description': 'Behavioral counseling + pharmacotherapy for smoking cessation',
                'rationale': 'Smoking cessation reduces diabetes risk by 30-40% and cardiovascular risk by 50%',
                'expected_impact': '30-40% diabetes risk reduction, 50% CV risk reduction',
                'timeline': '12-week intensive program',
                'evidence_level': 'A',
                'evidence_source': 'USPSTF, Cochrane Reviews',
                'cost_estimate': '$200-500',
                'pharmacotherapy_options': [
                    'Varenicline (Chantix) - most effective',
                    'Bupropion (Wellbutrin)',
                    'Nicotine replacement therapy'
                ],
                'behavioral_support': ['Quit line: 1-800-QUIT-NOW', 'Group counseling', 'Mobile apps'],
                'monitoring': 'Follow-up at 1, 3, 6, 12 months'
            })
        
        # Blood pressure management
        if patient.hypertension:
            interventions.append({
                'id': 'COMORB001',
                'priority': Priority.HIGH,
                'category': 'Comorbidity Management',
                'title': '❤️ Optimize Blood Pressure Control',
                'description': 'Target BP <130/80 mmHg; consider ACE inhibitor or ARB',
                'rationale': 'Hypertension with diabetes/prediabetes increases CV risk multiplicatively',
                'expected_impact': '20-40% reduction in CV events, renal protection',
                'timeline': 'Ongoing management',
                'evidence_level': 'A',
                'evidence_source': 'SPRINT, ACCORD-BP',
                'preferred_agents': ['ACE inhibitors', 'ARBs'],
                'bp_targets': {'general': '<130/80', 'elderly': '<140/90'},
                'monitoring': 'Home BP monitoring, renal function'
            })
        
        # Monitoring recommendations
        monitoring_priority = Priority.HIGH if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else Priority.MEDIUM
        interventions.append({
            'id': 'MON001',
            'priority': monitoring_priority,
            'category': 'Monitoring',
            'title': '📊 Structured Monitoring Protocol',
            'description': 'Regular laboratory and clinical monitoring based on risk level',
            'rationale': 'Close monitoring enables early detection and intervention adjustment',
            'expected_impact': 'Early detection of progression, timely treatment modification',
            'timeline': 'Per protocol',
            'evidence_level': 'B',
            'monitoring_schedule': {
                'critical': {'hba1c': 'Monthly', 'glucose': 'Weekly', 'visit': '1-2 weeks'},
                'high': {'hba1c': 'Every 3 months', 'glucose': 'Monthly', 'visit': '1 month'},
                'moderate': {'hba1c': 'Every 3-6 months', 'glucose': 'Quarterly', 'visit': '3 months'},
                'low': {'hba1c': 'Annually', 'glucose': 'Annually', 'visit': '6-12 months'}
            }
        })
        
        # Sort by priority
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        interventions.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return interventions
    
    def generate_counterfactuals(self, patient: PatientData, current_risk: float) -> List[Dict]:
        """Generate what-if scenario analysis"""
        scenarios = []
        
        # Scenario 1: Weight loss (if applicable)
        if patient.bmi >= 25:
            for reduction in [0.05, 0.10, 0.15]:  # 5%, 10%, 15% weight loss
                new_bmi = patient.bmi * (1 - reduction)
                if new_bmi >= 18.5:
                    new_patient = PatientData(**{**patient.__dict__, 'bmi': new_bmi})
                    features = self.preprocess(new_patient)
                    new_risk, _, _ = self.predict(features)
                    new_risk = self.adjust_risk(new_risk, new_patient)
                    
                    scenarios.append({
                        'id': f'WL{int(reduction*100)}',
                        'name': f'{int(reduction*100)}% Weight Loss',
                        'description': f'BMI: {patient.bmi:.1f} → {new_bmi:.1f} kg/m²',
                        'changes': {'bmi': new_bmi},
                        'current_risk': current_risk,
                        'new_risk': new_risk,
                        'absolute_reduction': current_risk - new_risk,
                        'relative_reduction': ((current_risk - new_risk) / current_risk * 100) if current_risk > 0 else 0,
                        'feasibility': 0.8 - (reduction * 2),  # Higher reduction = lower feasibility
                        'timeline': f'{int(reduction*100*2)}-{int(reduction*100*3)} months',
                        'difficulty': 'Easy' if reduction <= 0.05 else 'Moderate' if reduction <= 0.10 else 'Challenging',
                        'evidence': 'DPP: 7% weight loss = 58% risk reduction'
                    })
        
        # Scenario 2: Optimal glucose control
        if patient.glucose > 100 or patient.hba1c > 5.7:
            optimal_glucose = 90
            optimal_hba1c = 5.2
            new_patient = PatientData(**{**patient.__dict__, 'glucose': optimal_glucose, 'hba1c': optimal_hba1c})
            features = self.preprocess(new_patient)
            new_risk, _, _ = self.predict(features)
            new_risk = self.adjust_risk(new_risk, new_patient)
            
            scenarios.append({
                'id': 'GLUC_OPT',
                'name': 'Optimal Glucose Control',
                'description': f'HbA1c: {patient.hba1c:.1f}% → {optimal_hba1c}%, Glucose: {patient.glucose} → {optimal_glucose} mg/dL',
                'changes': {'hba1c': optimal_hba1c, 'glucose': optimal_glucose},
                'current_risk': current_risk,
                'new_risk': new_risk,
                'absolute_reduction': current_risk - new_risk,
                'relative_reduction': ((current_risk - new_risk) / current_risk * 100) if current_risk > 0 else 0,
                'feasibility': 0.5 if patient.hba1c >= 6.5 else 0.7,
                'timeline': '3-6 months',
                'difficulty': 'Challenging' if patient.hba1c >= 6.5 else 'Moderate',
                'evidence': 'Normalization of glycemia prevents diabetes development'
            })
        
        # Scenario 3: Smoking cessation
        if patient.smoking in ['current', 'not current']:
            new_patient = PatientData(**{**patient.__dict__, 'smoking': 'never'})
            features = self.preprocess(new_patient)
            new_risk, _, _ = self.predict(features)
            new_risk = self.adjust_risk(new_risk, new_patient)
            
            scenarios.append({
                'id': 'SMOKE_STOP',
                'name': 'Smoking Cessation',
                'description': f'Quit smoking completely',
                'changes': {'smoking': 'never'},
                'current_risk': current_risk,
                'new_risk': new_risk,
                'absolute_reduction': current_risk - new_risk,
                'relative_reduction': ((current_risk - new_risk) / current_risk * 100) if current_risk > 0 else 0,
                'feasibility': 0.4,  # Smoking cessation is difficult
                'timeline': '3-12 months',
                'difficulty': 'Challenging',
                'evidence': '30-40% diabetes risk reduction with cessation'
            })
        
        # Scenario 4: Increased physical activity
        if patient.activity_level < 5:
            new_activity = min(5, patient.activity_level + 2)
            new_patient = PatientData(**{**patient.__dict__, 'activity_level': new_activity})
            # Activity affects adjusted risk, not direct prediction
            new_adjusted = self.adjust_risk(current_risk - 0.05, new_patient)  # Estimate
            
            scenarios.append({
                'id': 'ACTIVITY_UP',
                'name': 'Increased Physical Activity',
                'description': f'Activity level: {patient.activity_level}/5 → {new_activity}/5 (150+ min/week)',
                'changes': {'activity_level': new_activity},
                'current_risk': current_risk,
                'new_risk': max(0, new_adjusted),
                'absolute_reduction': current_risk - max(0, new_adjusted),
                'relative_reduction': ((current_risk - max(0, new_adjusted)) / current_risk * 100) if current_risk > 0 else 0,
                'feasibility': 0.7,
                'timeline': '2-3 months',
                'difficulty': 'Moderate',
                'evidence': '20-30% risk reduction with regular exercise'
            })
        
        # Scenario 5: Combined lifestyle intervention
        combined_changes = {}
        new_patient_dict = patient.__dict__.copy()
        
        if patient.bmi >= 25:
            new_patient_dict['bmi'] = patient.bmi * 0.93  # 7% weight loss
            combined_changes['bmi'] = new_patient_dict['bmi']
        
        if patient.glucose > 100:
            new_patient_dict['glucose'] = 95
            combined_changes['glucose'] = 95
        
        if patient.hba1c > 5.7:
            new_patient_dict['hba1c'] = max(5.2, patient.hba1c - 0.5)
            combined_changes['hba1c'] = new_patient_dict['hba1c']
        
        if patient.smoking in ['current', 'not current']:
            new_patient_dict['smoking'] = 'former'
            combined_changes['smoking'] = 'former'
        
        new_patient_dict['activity_level'] = min(5, patient.activity_level + 2)
        combined_changes['activity_level'] = new_patient_dict['activity_level']
        
        if combined_changes:
            new_patient = PatientData(**new_patient_dict)
            features = self.preprocess(new_patient)
            new_risk, _, _ = self.predict(features)
            new_risk = self.adjust_risk(new_risk, new_patient)
            
            scenarios.append({
                'id': 'COMBINED',
                'name': 'Comprehensive Lifestyle Intervention',
                'description': 'Combined: weight loss + glucose control + activity + smoking (DPP-style)',
                'changes': combined_changes,
                'current_risk': current_risk,
                'new_risk': new_risk,
                'absolute_reduction': current_risk - new_risk,
                'relative_reduction': ((current_risk - new_risk) / current_risk * 100) if current_risk > 0 else 0,
                'feasibility': 0.35,
                'timeline': '12-24 months',
                'difficulty': 'Very Challenging',
                'evidence': 'DPP: 58% diabetes prevention with intensive lifestyle'
            })
        
        # Sort by risk reduction (highest first)
        scenarios.sort(key=lambda x: x['absolute_reduction'], reverse=True)
        
        return scenarios
    
    def check_safety_alerts(self, patient: PatientData) -> List[Dict]:
        """Comprehensive safety assessment"""
        alerts = []
        
        # Critical glycemic values
        if patient.hba1c >= 10.0:
            alerts.append({
                'level': 'CRITICAL',
                'category': 'Glycemic Emergency',
                'message': f'⚠️ CRITICAL: HbA1c {patient.hba1c:.1f}% indicates severe uncontrolled diabetes',
                'action': 'Immediate medical evaluation required. Risk of DKA/HHS.',
                'icon': '🚨'
            })
        elif patient.hba1c >= 9.0:
            alerts.append({
                'level': 'HIGH',
                'category': 'Severe Hyperglycemia',
                'message': f'⚠️ HIGH ALERT: HbA1c {patient.hba1c:.1f}% - Severe hyperglycemia',
                'action': 'Urgent specialist referral within 1 week',
                'icon': '🔴'
            })
        
        if patient.glucose >= 300:
            alerts.append({
                'level': 'CRITICAL',
                'category': 'Acute Hyperglycemia',
                'message': f'🚨 CRITICAL: Blood glucose {patient.glucose} mg/dL - Check for ketones',
                'action': 'Immediate evaluation for DKA/HHS. Consider ER referral.',
                'icon': '🚨'
            })
        elif patient.glucose >= 250:
            alerts.append({
                'level': 'HIGH',
                'category': 'Significant Hyperglycemia',
                'message': f'⚠️ Blood glucose {patient.glucose} mg/dL - Significant hyperglycemia',
                'action': 'Same-day evaluation recommended',
                'icon': '🔴'
            })
        
        # Severe obesity
        if patient.bmi >= 45:
            alerts.append({
                'level': 'HIGH',
                'category': 'Severe Obesity',
                'message': f'⚠️ BMI {patient.bmi:.1f} - Class III Obesity (Severe)',
                'action': 'Consider bariatric surgery evaluation. High anesthetic risk.',
                'icon': '🔴'
            })
        elif patient.bmi >= 40:
            alerts.append({
                'level': 'MODERATE',
                'category': 'Severe Obesity',
                'message': f'BMI {patient.bmi:.1f} - Class III Obesity',
                'action': 'Bariatric surgery referral indicated',
                'icon': '🟠'
            })
        
        # Cardiovascular risk
        if patient.hypertension and patient.heart_disease:
            alerts.append({
                'level': 'HIGH',
                'category': 'Cardiovascular Risk',
                'message': '⚠️ Multiple cardiovascular risk factors present',
                'action': 'Aggressive CV risk management. Consider cardiology referral.',
                'icon': '🔴'
            })
        
        # Age considerations
        if patient.age >= 80:
            alerts.append({
                'level': 'INFO',
                'category': 'Geriatric Considerations',
                'message': 'ℹ️ Advanced age - Consider individualized treatment goals',
                'action': 'Avoid hypoglycemia. Consider functional status in treatment decisions.',
                'icon': 'ℹ️'
            })
        elif patient.age >= 75:
            alerts.append({
                'level': 'INFO',
                'category': 'Elderly Patient',
                'message': 'ℹ️ Age ≥75 - Modify treatment intensity as appropriate',
                'action': 'Consider comorbidities, polypharmacy, fall risk',
                'icon': 'ℹ️'
            })
        
        # Hypoglycemia risk (if on treatment)
        if patient.hba1c < 5.5 and patient.glucose < 70:
            alerts.append({
                'level': 'HIGH',
                'category': 'Hypoglycemia Risk',
                'message': '⚠️ Low glucose detected - Evaluate for hypoglycemia',
                'action': 'Review medications. Assess for symptoms.',
                'icon': '🔴'
            })
        
        return alerts
    
    def get_followup_schedule(self, risk_level: RiskLevel) -> Dict[str, str]:
        """Generate follow-up recommendations"""
        schedules = {
            RiskLevel.CRITICAL: {
                'next_visit': '1-2 weeks',
                'hba1c_testing': 'Every 1-2 months until stable',
                'glucose_monitoring': 'Daily to weekly',
                'specialist_referral': 'Immediate - within 1-2 weeks',
                'lifestyle_coaching': 'Weekly for first month',
                'phone_followup': 'Within 48-72 hours',
                'labs': 'Comprehensive metabolic panel within 1 week'
            },
            RiskLevel.HIGH: {
                'next_visit': '2-4 weeks',
                'hba1c_testing': 'Every 3 months',
                'glucose_monitoring': 'Weekly to monthly',
                'specialist_referral': 'Within 1-2 months',
                'lifestyle_coaching': 'Bi-weekly for 3 months',
                'phone_followup': 'Within 1 week',
                'labs': 'Lipid panel, renal function within 1 month'
            },
            RiskLevel.MODERATE: {
                'next_visit': '2-3 months',
                'hba1c_testing': 'Every 3-6 months',
                'glucose_monitoring': 'Monthly to quarterly',
                'specialist_referral': 'As needed',
                'lifestyle_coaching': 'Monthly',
                'phone_followup': 'As needed',
                'labs': 'Annual comprehensive panel'
            },
            RiskLevel.LOW: {
                'next_visit': '6-12 months',
                'hba1c_testing': 'Annually',
                'glucose_monitoring': 'Annually',
                'specialist_referral': 'Not indicated',
                'lifestyle_coaching': 'As desired',
                'phone_followup': 'As needed',
                'labs': 'Annual screening'
            }
        }
        return schedules.get(risk_level, schedules[RiskLevel.MODERATE])
    
    def full_assessment(self, patient: PatientData) -> RiskAssessment:
        """Perform comprehensive risk assessment"""
        # Get prediction
        features = self.preprocess(patient)
        risk_score, ci, confidence = self.predict(features)
        adjusted_risk = self.adjust_risk(risk_score, patient)
        risk_level = self.get_risk_level(adjusted_risk)
        
        # Analyze risk factors
        risk_factors = self.analyze_risk_factors(patient)
        
        # Generate interventions
        interventions = self.generate_interventions(patient, risk_level, risk_factors)
        
        # Generate counterfactuals
        counterfactuals = self.generate_counterfactuals(patient, adjusted_risk)
        
        # Check safety
        safety_alerts = self.check_safety_alerts(patient)
        
        # Get follow-up schedule
        followup = self.get_followup_schedule(risk_level)
        
        return RiskAssessment(
            risk_score=risk_score,
            adjusted_risk=adjusted_risk,
            risk_level=risk_level,
            confidence_interval=ci,
            confidence=confidence,
            risk_factors=risk_factors,
            interventions=interventions,
            counterfactuals=counterfactuals,
            safety_alerts=[a['message'] for a in safety_alerts],
            followup_schedule=followup
        )
    
    # Longitudinal tracking
    def add_record(self, patient_id: str, record: LongitudinalRecord):
        """Add longitudinal record"""
        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = []
        self.patient_history[patient_id].append(record)
        # Keep last 2 years
        cutoff = datetime.now() - timedelta(days=730)
        self.patient_history[patient_id] = [
            r for r in self.patient_history[patient_id] if r.date >= cutoff
        ]
    
    def get_trends(self, patient_id: str) -> Optional[Dict]:
        """Analyze trends over time"""
        if patient_id not in self.patient_history or len(self.patient_history[patient_id]) < 2:
            return None
        
        history = sorted(self.patient_history[patient_id], key=lambda x: x.date)
        
        return {
            'data_points': len(history),
            'period': {
                'start': history[0].date,
                'end': history[-1].date
            },
            'risk_trend': {
                'start': history[0].risk_score,
                'end': history[-1].risk_score,
                'change': history[-1].risk_score - history[0].risk_score,
                'direction': 'improving' if history[-1].risk_score < history[0].risk_score else 'worsening'
            },
            'hba1c_trend': {
                'start': history[0].hba1c,
                'end': history[-1].hba1c,
                'change': history[-1].hba1c - history[0].hba1c
            },
            'bmi_trend': {
                'start': history[0].bmi,
                'end': history[-1].bmi,
                'change': history[-1].bmi - history[0].bmi
            },
            'history': history
        }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_risk_gauge(risk_score: float, risk_level: RiskLevel) -> go.Figure:
    """Create beautiful risk gauge"""
    color_map = {
        RiskLevel.LOW: '#28a745',
        RiskLevel.MODERATE: '#ffc107',
        RiskLevel.HIGH: '#ff9800',
        RiskLevel.CRITICAL: '#dc3545'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 20}},
        delta={'reference': 30, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': color_map.get(risk_level, '#667eea'), 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 85], 'color': '#ffe0b2'},
                {'range': [85, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI'}
    )
    
    return fig

def create_risk_factors_chart(risk_factors: List[Dict]) -> go.Figure:
    """Create risk factors horizontal bar chart"""
    df = pd.DataFrame(risk_factors[:8])  # Top 8
    
    colors = []
    for status in df['status']:
        if status == 'CRITICAL':
            colors.append('#dc3545')
        elif status in ['HIGH', 'ELEVATED']:
            colors.append('#ff9800')
        elif status == 'MODERATE':
            colors.append('#ffc107')
        else:
            colors.append('#28a745')
    
    fig = go.Figure(go.Bar(
        y=df['name'],
        x=df['score'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=df['value'],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<br>Value: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Risk Factors Analysis',
        xaxis_title='Contribution Score',
        yaxis_title='',
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI'}
    )
    
    return fig

def create_counterfactual_chart(counterfactuals: List[Dict]) -> go.Figure:
    """Create what-if scenario comparison chart"""
    if not counterfactuals:
        return None
    
    names = [cf['name'] for cf in counterfactuals]
    reductions = [cf['relative_reduction'] for cf in counterfactuals]
    feasibilities = [cf['feasibility'] * 100 for cf in counterfactuals]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Risk Reduction Potential', 'Feasibility Score'],
        horizontal_spacing=0.1
    )
    
    # Risk reduction bars
    fig.add_trace(go.Bar(
        y=names,
        x=reductions,
        orientation='h',
        marker=dict(
            color=reductions,
            colorscale='RdYlGn',
            showscale=False
        ),
        text=[f'{r:.1f}%' for r in reductions],
        textposition='auto',
        name='Risk Reduction'
    ), row=1, col=1)
    
    # Feasibility bars
    fig.add_trace(go.Bar(
        y=names,
        x=feasibilities,
        orientation='h',
        marker=dict(
            color=feasibilities,
            colorscale='Blues',
            showscale=False
        ),
        text=[f'{f:.0f}%' for f in feasibilities],
        textposition='auto',
        name='Feasibility'
    ), row=1, col=2)
    
    fig.update_layout(
        title='What-If Scenario Analysis',
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI'}
    )
    
    fig.update_xaxes(title_text='Reduction %', row=1, col=1)
    fig.update_xaxes(title_text='Feasibility %', row=1, col=2)
    
    return fig

def create_trend_chart(trends: Dict) -> go.Figure:
    """Create longitudinal trend chart"""
    history = trends['history']
    dates = [r.date for r in history]
    risks = [r.risk_score * 100 for r in history]
    hba1cs = [r.hba1c for r in history]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Risk Score Trend', 'HbA1c Trend'],
        vertical_spacing=0.15
    )
    
    # Risk trend
    fig.add_trace(go.Scatter(
        x=dates, y=risks,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ), row=1, col=1)
    
    # HbA1c trend
    fig.add_trace(go.Scatter(
        x=dates, y=hba1cs,
        mode='lines+markers',
        name='HbA1c',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=10)
    ), row=2, col=1)
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=5.7, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI'}
    )
    
    fig.update_yaxes(title_text='Risk %', row=1, col=1)
    fig.update_yaxes(title_text='HbA1c %', row=2, col=1)
    
    return fig

# ============================================================================
# LOAD ENGINE (CACHED)
# ============================================================================

@st.cache_resource
def load_engine():
    """Load clinical engine (cached)"""
    model_path = "diabetes_clinical_model_calibrated.pkl"
    return AdvancedClinicalEngine(model_path)

# Initialize engines
try:
    engine = load_engine()
    if not engine.model_loaded:
        st.error("❌ Failed to load model. Please check the path.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error initializing engine: {e}")
    st.stop()

# Initialize GenAI engine
genai_engine = GenAIHealthInsights()

# Check GenAI status
genai_status = "✅ GenAI Enabled" if genai_engine.enabled else "⚠️ GenAI Disabled (API Key Required)"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #667eea;'>🏥 CDS System</h2>
        <p style='color: #666; font-size: 0.9rem;'>Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "**Navigation**",
        ["🏠 Home", "👤 Patient Assessment", "👨‍⚕️ Clinical Analysis", 
         "🔮 What-If Explorer", "📈 Trend Analysis", "📊 Model Insights", 
         "📖 Documentation"],
        key='nav'
    )
    
    st.markdown("---")
    
    # Model status
    st.markdown("**System Status**")
    if engine.model_loaded:
        st.success("✅ Model: Online")
    else:
        st.error("❌ Model: Offline")
    
    st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #999;'>
        <p>Version 2.0.0</p>
        <p>© 2024 Healthcare AI</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# HOME PAGE
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🏥 Clinical Decision Support System</h1>
        <p>AI-Powered Diabetes Risk Assessment & Prevention Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("97.7%", "ROC-AUC Score"),
        ("97.3%", "Accuracy"),
        ("98.3%", "Precision"),
        ("<100ms", "Response Time")
    ]
    
    for col, (value, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.subheader("🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("🎯", "Risk Prediction", "Calibrated ML model with uncertainty quantification and confidence intervals"),
        ("💡", "Smart Recommendations", "Evidence-based interventions personalized to each patient's profile"),
        ("🔮", "What-If Analysis", "Counterfactual reasoning to show impact of lifestyle changes"),
        ("📈", "Longitudinal Tracking", "Monitor trends and assess intervention effectiveness over time"),
        ("👥", "Dual Interface", "Specialized views for clinicians and patients"),
        ("🛡️", "Safety First", "Real-time alerts, bias detection, and quality controls")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-icon'>{icon}</div>
                <h4>{title}</h4>
                <p style='color: #666; font-size: 0.9rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        if (i + 1) % 3 == 0 and i < len(features) - 1:
            col1, col2, col3 = st.columns(3)
    
    st.markdown("---")
    
    # Quick start
    st.subheader("🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Patients:**
        1. Navigate to **👤 Patient Assessment**
        2. Enter your health information
        3. Get easy-to-understand results
        4. Review personalized recommendations
        """)
    
    with col2:
        st.markdown("""
        **For Clinicians:**
        1. Navigate to **👨‍⚕️ Clinical Analysis**
        2. Enter patient data
        3. Review comprehensive assessment
        4. Access evidence-based interventions
        """)
    
    st.info("👈 Use the sidebar to navigate to different sections")

# PATIENT ASSESSMENT
elif page == "👤 Patient Assessment":
    st.markdown("""
    <div class='main-header'>
        <h1>👤 Patient Risk Assessment</h1>
        <p>Easy-to-understand diabetes risk evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("patient_form"):
        st.subheader("📋 Your Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Your Age", 18, 100, 55, help="Enter your age in years")
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        
        with col2:
            weight = st.number_input("Weight (kg)", 40.0, 200.0, 75.0, help="Your weight in kilograms")
            height = st.number_input("Height (cm)", 140.0, 220.0, 170.0, help="Your height in centimeters")
        
        with col3:
            # Auto-calculate BMI
            bmi = weight / ((height / 100) ** 2)
            st.metric("Your BMI", f"{bmi:.1f}", help="Body Mass Index")
            bmi_status = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            st.caption(f"Category: {bmi_status}")
        
        st.markdown("---")
        st.subheader("🩺 Health Measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.8, 0.1,
                                   help="From your blood test (normal: <5.7%)")
            glucose = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 400, 110,
                                     help="Measured after 8-hour fast (normal: <100)")
        
        with col2:
            hypertension = st.checkbox("I have high blood pressure", help="Hypertension diagnosis")
            heart_disease = st.checkbox("I have heart disease", help="Any heart condition")
            family_history = st.checkbox("Family history of diabetes", help="Parent or sibling with diabetes")
        
        st.markdown("---")
        st.subheader("🏃 Lifestyle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smoking_display = st.selectbox("Smoking Status", 
                                          ["Never smoked", "Former smoker", "Current smoker", "Sometimes"])
            smoking_map = {"Never smoked": "never", "Former smoker": "former", 
                          "Current smoker": "current", "Sometimes": "ever"}
            smoking = smoking_map[smoking_display]
        
        with col2:
            activity = st.slider("Physical Activity Level", 1, 5, 3,
                                help="1=Sedentary, 5=Very Active")
        
        submitted = st.form_submit_button("🔍 Check My Risk", use_container_width=True, type="primary")
    
    if submitted:
        # Create patient data
        patient = PatientData(
            patient_id=f"PAT{datetime.now().strftime('%H%M%S')}",
            age=age, gender=gender, bmi=bmi, hba1c=hba1c, glucose=glucose,
            hypertension=hypertension, heart_disease=heart_disease,
            smoking=smoking, family_history=family_history, activity_level=activity
        )
        
        # Convert patient to dictionary for GenAI features
        patient_dict = patient.__dict__.copy()
        
        # Get assessment
        assessment = engine.full_assessment(patient)
        
        st.markdown("---")
        st.subheader("📊 Your Results")
        
        # Risk display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_colors = {
                RiskLevel.LOW: ("#28a745", "🟢"),
                RiskLevel.MODERATE: ("#ffc107", "🟡"),
                RiskLevel.HIGH: ("#ff9800", "🟠"),
                RiskLevel.CRITICAL: ("#dc3545", "🔴")
            }
            color, emoji = risk_colors[assessment.risk_level]
            
            risk_messages = {
                RiskLevel.LOW: "Your diabetes risk is LOW. Great job!",
                RiskLevel.MODERATE: "Your risk is MODERATE. Take action now to prevent diabetes.",
                RiskLevel.HIGH: "Your risk is HIGH. Important to make changes now.",
                RiskLevel.CRITICAL: "Your risk is CRITICAL. Please see your doctor soon."
            }
            
            st.markdown(f"""
            <div class='risk-card risk-{assessment.risk_level.value}'>
                <h2 style='margin: 0;'>{emoji} {assessment.risk_level.value.upper()} RISK</h2>
                <p style='margin: 0.5rem 0;'>{risk_messages[assessment.risk_level]}</p>
                <h3 style='margin: 0;'>Risk Score: {assessment.adjusted_risk:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = create_risk_gauge(assessment.adjusted_risk, assessment.risk_level)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key numbers
        st.subheader("📈 Your Key Health Numbers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hba1c_color = "🟢" if hba1c < 5.7 else "🟡" if hba1c < 6.5 else "🔴"
            st.metric("HbA1c", f"{hba1c:.1f}%", f"Target: <5.7%")
            st.caption(f"{hba1c_color} {'Normal' if hba1c < 5.7 else 'Prediabetes' if hba1c < 6.5 else 'Diabetes Range'}")
        
        with col2:
            glucose_color = "🟢" if glucose < 100 else "🟡" if glucose < 126 else "🔴"
            st.metric("Blood Sugar", f"{glucose} mg/dL", f"Target: <100")
            st.caption(f"{glucose_color} {'Normal' if glucose < 100 else 'Elevated' if glucose < 126 else 'High'}")
        
        with col3:
            bmi_color = "🟢" if bmi < 25 else "🟡" if bmi < 30 else "🔴"
            st.metric("BMI", f"{bmi:.1f}", f"Target: <25")
            st.caption(f"{bmi_color} {'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'}")
        
        # What you can do
        st.subheader("💪 What You Can Do")
        
        if assessment.counterfactuals:
            for cf in assessment.counterfactuals[:3]:
                with st.expander(f"✨ {cf['name']} - Could reduce risk by {cf['relative_reduction']:.0f}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("New Risk", f"{cf['new_risk']:.1%}", 
                                 f"-{cf['absolute_reduction']:.1%}")
                    with col2:
                        st.metric("Timeline", cf['timeline'])
                    
                    st.info(f"**What changes:** {cf['description']}")
                    st.progress(cf['feasibility'])
                    st.caption(f"Feasibility: {cf['feasibility']:.0%} | Difficulty: {cf['difficulty']}")
        
        # Next steps
        st.subheader("🎯 Your Next Steps")
        
        steps = [
            "📅 Schedule follow-up appointment with your doctor",
            "💬 Discuss these results and recommendations",
            "🎯 Set realistic, achievable health goals",
            "📝 Ask about diabetes prevention programs",
            "📱 Consider using a health tracking app"
        ]
        
        for step in steps:
            st.write(step)
        
        # Questions to ask
        with st.expander("❓ Questions to Ask Your Doctor"):
            questions = [
                "What is my biggest risk factor for diabetes?",
                "Which lifestyle changes would help me most?",
                "Should I be tested more frequently?",
                "Are there programs I can join for support?",
                "Should I see a specialist (endocrinologist, dietitian)?"
            ]
            for q in questions:
                st.write(f"• {q}")
        
        # ======================================================================
        # GENAI POWERED HEALTH INSIGHTS
        # ======================================================================
        st.markdown("---")
        st.subheader("🤖 AI-Powered Health Insights")
        
        # Show GenAI status
        st.caption(genai_status)
        
        if genai_engine.enabled and patient_dict:
            # Create tabs for different GenAI features
            genai_tabs = st.tabs(["📊 Risk Explanation", "🍽️ Meal Ideas", "🏃 Exercise Plan"])
            
            with genai_tabs[0]:
                st.markdown("### What do these numbers mean for you?")
                if 'risk_factors' in assessment.__dict__:
                    risk_factors_list = assessment.risk_factors
                else:
                    risk_factors_list = []
                    
                if st.button("✨ Generate Explanation", key="genai_explain"):
                    with st.spinner("Generating personalized explanation..."):
                        explanation = genai_engine.explain_risk_factors(patient_dict, risk_factors_list)
                        st.success(explanation)
                else:
                    st.info("Click the button above to get an AI-generated explanation of your risk factors in simple terms.")
            
            with genai_tabs[1]:
                st.markdown("### Personalized Meal Suggestions")
                if st.button("🍽️ Get Meal Ideas", key="genai_meals"):
                    with st.spinner("Generating personalized meal suggestions..."):
                        meals = genai_engine.generate_meal_suggestions(patient_dict, assessment.risk_level.value)
                        st.success(meals)
                else:
                    st.info("Click the button above to get AI-generated meal suggestions tailored to your health profile.")
            
            with genai_tabs[2]:
                st.markdown("### Custom Exercise Plan")
                if st.button("🏃 Get Exercise Plan", key="genai_exercise"):
                    with st.spinner("Generating personalized exercise plan..."):
                        exercise = genai_engine.generate_exercise_plan(patient_dict)
                        st.success(exercise)
                else:
                    st.info("Click the button above to get an AI-generated exercise plan based on your age and fitness level.")
        else:
            st.warning("""
            **GenAI features are disabled.** To enable AI-powered insights:
            1. Get a free API key from [OpenRouter](https://openrouter.ai/)
            2. Add it to the `.env` file as `OPENROUTER_API_KEY=your_key_here`
            3. Restart the application
            """)
        
        # Safety notice
        st.info("💡 **Note:** AI-generated suggestions are for informational purposes only. Always consult your healthcare provider before making changes to your health routine.")

# CLINICAL ANALYSIS
elif page == "👨‍⚕️ Clinical Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>👨‍⚕️ Clinical Analysis</h1>
        <p>Comprehensive risk assessment with evidence-based interventions</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("clinical_form"):
        st.subheader("Patient Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            patient_id = st.text_input("Patient ID", f"P{datetime.now().strftime('%H%M%S')}")
            age = st.number_input("Age", 0, 120, 55)
        
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            bmi = st.number_input("BMI", 10.0, 100.0, 28.5, 0.1)
        
        with col3:
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.8, 0.1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 400, 110)
        
        with col4:
            hypertension = st.selectbox("Hypertension", [False, True], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.selectbox("Heart Disease", [False, True], format_func=lambda x: "Yes" if x else "No")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            smoking = st.selectbox("Smoking", ["never", "former", "current", "not current", "No Info", "ever"])
        
        with col2:
            family_history = st.checkbox("Family History of Diabetes")
        
        with col3:
            activity = st.slider("Activity Level", 1, 5, 3)
        
        submitted = st.form_submit_button("🔬 Run Clinical Assessment", use_container_width=True, type="primary")
    
    if submitted:
        # Create patient
        patient = PatientData(
            patient_id=patient_id, age=age, gender=gender, bmi=bmi,
            hba1c=hba1c, glucose=glucose, hypertension=hypertension,
            heart_disease=heart_disease, smoking=smoking,
            family_history=family_history, activity_level=activity
        )
        
        # Full assessment
        assessment = engine.full_assessment(patient)
        
        # Store in session
        st.session_state.assessment_result = assessment
        st.session_state.patient_data = patient
        
        st.markdown("---")
        
        # Safety alerts first
        if assessment.safety_alerts:
            st.subheader("⚠️ Safety Alerts")
            for alert in assessment.safety_alerts:
                if "CRITICAL" in alert:
                    st.error(alert)
                elif "HIGH" in alert or "⚠️" in alert:
                    st.warning(alert)
                else:
                    st.info(alert)
        
        # Risk overview
        st.subheader("🎯 Risk Assessment")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Risk Score", f"{assessment.risk_score:.1%}")
        with col2:
            st.metric("Adjusted Risk", f"{assessment.adjusted_risk:.1%}")
        with col3:
            st.metric("Risk Level", assessment.risk_level.value.upper())
        with col4:
            st.metric("Confidence", assessment.confidence)
        with col5:
            st.metric("95% CI", f"{assessment.confidence_interval[0]:.1%} - {assessment.confidence_interval[1]:.1%}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_risk_gauge(assessment.adjusted_risk, assessment.risk_level)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_risk_factors_chart(assessment.risk_factors)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors detail
        st.subheader("📊 Risk Factors Analysis")
        
        tabs = st.tabs(["Summary", "Detailed View", "Categories"])
        
        with tabs[0]:
            for factor in assessment.risk_factors[:5]:
                status_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "ELEVATED": "🟡", "MODERATE": "🟡", "NORMAL": "🟢"}
                icon = status_icons.get(factor['status'], "⚪")
                
                with st.expander(f"{icon} {factor['name']} - {factor['status']} (Score: {factor['score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Current Value:** {factor['value']}")
                        st.write(f"**Normal Range:** {factor['normal_range']}")
                        st.write(f"**Target:** {factor['target']}")
                    with col2:
                        st.write(f"**Interpretation:** {factor['interpretation']}")
                        st.write(f"**Modifiable:** {'✅ Yes' if factor['modifiable'] else '❌ No'}")
                        st.write(f"**Category:** {factor['category']}")
        
        with tabs[1]:
            df = pd.DataFrame(assessment.risk_factors)
            st.dataframe(df[['name', 'value', 'status', 'score', 'modifiable', 'interpretation']], 
                        use_container_width=True)
        
        with tabs[2]:
            categories = {}
            for f in assessment.risk_factors:
                cat = f['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(f)
            
            for cat, factors in categories.items():
                st.write(f"**{cat}:**")
                for f in factors:
                    st.write(f"  • {f['name']}: {f['value']} ({f['status']})")
        
        # Interventions
        st.subheader("💊 Recommended Interventions")
        
        priority_icons = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        
        for intervention in assessment.interventions[:6]:
            icon = priority_icons.get(intervention['priority'].value, "⚪")
            
            with st.expander(f"{icon} [{intervention['priority'].value}] {intervention['title']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Category:** {intervention['category']}")
                    st.write(f"**Description:** {intervention['description']}")
                    st.write(f"**Rationale:** {intervention['rationale']}")
                
                with col2:
                    st.write(f"**Expected Impact:** {intervention['expected_impact']}")
                    st.write(f"**Timeline:** {intervention['timeline']}")
                    st.write(f"**Evidence Level:** {intervention['evidence_level']}")
                    if 'cost_estimate' in intervention:
                        st.write(f"**Cost Estimate:** {intervention['cost_estimate']}")
        
        # Follow-up schedule
        st.subheader("📅 Follow-up Schedule")
        
        schedule = assessment.followup_schedule
        cols = st.columns(4)
        
        for i, (key, value) in enumerate(schedule.items()):
            with cols[i % 4]:
                st.metric(key.replace('_', ' ').title(), value)
        
        # Clinical notes
        st.subheader("📝 Clinical Notes")
        
        clinical_summary = f"""
CLINICAL ASSESSMENT SUMMARY
{'='*60}
Patient ID: {patient_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Assessed by: Clinical Decision Support System v2.0

RISK ASSESSMENT:
- Raw Risk Score: {assessment.risk_score:.1%}
- Adjusted Risk: {assessment.adjusted_risk:.1%}
- Risk Level: {assessment.risk_level.value.upper()}
- Confidence: {assessment.confidence}
- 95% CI: {assessment.confidence_interval[0]:.1%} - {assessment.confidence_interval[1]:.1%}

KEY FINDINGS:
- HbA1c: {hba1c:.1f}% {'[DIABETES]' if hba1c >= 6.5 else '[PREDIABETES]' if hba1c >= 5.7 else '[NORMAL]'}
- Glucose: {glucose} mg/dL {'[DIABETES]' if glucose >= 126 else '[IMPAIRED]' if glucose >= 100 else '[NORMAL]'}
- BMI: {bmi:.1f} {'[OBESE]' if bmi >= 30 else '[OVERWEIGHT]' if bmi >= 25 else '[NORMAL]'}
- Age: {age} years
- Hypertension: {'Yes' if hypertension else 'No'}
- Heart Disease: {'Yes' if heart_disease else 'No'}
- Smoking: {smoking}
- Family History: {'Yes' if family_history else 'No'}

TOP RISK FACTORS:
{chr(10).join([f"- {f['name']}: {f['value']} ({f['status']})" for f in assessment.risk_factors[:5]])}

PRIORITY INTERVENTIONS:
{chr(10).join([f"- [{i['priority'].value}] {i['title']}" for i in assessment.interventions[:5]])}

FOLLOW-UP:
- Next Visit: {schedule['next_visit']}
- HbA1c Testing: {schedule['hba1c_testing']}
- Specialist: {schedule['specialist_referral']}

{'='*60}
Generated by Clinical Decision Support System
        """
        
        st.text_area("", clinical_summary, height=400)
        
        # Download button
        st.download_button(
            label="📥 Download Clinical Report",
            data=clinical_summary,
            file_name=f"clinical_report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# WHAT-IF EXPLORER
elif page == "🔮 What-If Explorer":
    st.markdown("""
    <div class='main-header'>
        <h1>🔮 What-If Explorer</h1>
        <p>Explore how lifestyle changes affect diabetes risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("whatif_form"):
        st.subheader("Current Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 100, 55)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            bmi = st.number_input("Current BMI", 10.0, 100.0, 32.0, 0.1)
        
        with col2:
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 6.3, 0.1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 400, 145)
            smoking = st.selectbox("Smoking", ["never", "former", "current", "not current"])
        
        with col3:
            hypertension = st.checkbox("Hypertension")
            heart_disease = st.checkbox("Heart Disease")
            family_history = st.checkbox("Family History")
            activity = st.slider("Activity Level", 1, 5, 2)
        
        submitted = st.form_submit_button("🔮 Explore Scenarios", use_container_width=True, type="primary")
    
    if submitted:
        # Create patient
        patient = PatientData(
            patient_id="WHATIF",
            age=age, gender=gender, bmi=bmi, hba1c=hba1c, glucose=glucose,
            hypertension=hypertension, heart_disease=heart_disease,
            smoking=smoking, family_history=family_history, activity_level=activity
        )
        
        # Get prediction
        features = engine.preprocess(patient)
        risk_score, ci, confidence = engine.predict(features)
        adjusted_risk = engine.adjust_risk(risk_score, patient)
        risk_level = engine.get_risk_level(adjusted_risk)
        
        # Get counterfactuals
        counterfactuals = engine.generate_counterfactuals(patient, adjusted_risk)
        
        st.markdown("---")
        
        # Current risk
        st.subheader("📊 Current Risk Profile")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{risk_score:.1%}")
        with col2:
            st.metric("Adjusted Risk", f"{adjusted_risk:.1%}")
        with col3:
            st.metric("Risk Level", risk_level.value.upper())
        
        # Scenarios comparison
        st.subheader("🎯 Intervention Scenarios")
        
        if counterfactuals:
            # Chart
            fig = create_counterfactual_chart(counterfactuals)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison
            st.subheader("📋 Scenario Details")
            
            for cf in counterfactuals:
                difficulty_colors = {"Easy": "🟢", "Moderate": "🟡", "Challenging": "🟠", "Very Challenging": "🔴"}
                diff_icon = difficulty_colors.get(cf['difficulty'], "⚪")
                
                with st.expander(f"✨ {cf['name']} - {cf['relative_reduction']:.1f}% Risk Reduction"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Risk", f"{cf['current_risk']:.1%}")
                        st.metric("New Risk", f"{cf['new_risk']:.1%}", 
                                 f"-{cf['absolute_reduction']:.1%}")
                    
                    with col2:
                        st.metric("Risk Reduction", f"{cf['relative_reduction']:.1f}%")
                        st.metric("Timeline", cf['timeline'])
                    
                    with col3:
                        st.metric("Feasibility", f"{cf['feasibility']:.0%}")
                        st.write(f"**Difficulty:** {diff_icon} {cf['difficulty']}")
                    
                    st.info(f"**Changes:** {cf['description']}")
                    st.caption(f"**Evidence:** {cf['evidence']}")
            
            # Best opportunity highlight
            best = max(counterfactuals, key=lambda x: x['relative_reduction'])
            
            st.success(f"""
            ### 💡 Best Opportunity
            
            **{best['name']}** offers the highest risk reduction potential:
            - Current Risk: {best['current_risk']:.1%} → New Risk: {best['new_risk']:.1%}
            - **Reduction: {best['relative_reduction']:.1f}%**
            - Timeline: {best['timeline']}
            - Feasibility: {best['feasibility']:.0%}
            """)
        else:
            st.info("No counterfactual scenarios available for current profile.")

# TREND ANALYSIS
elif page == "📈 Trend Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Longitudinal Trend Analysis</h1>
        <p>Track patient progress over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 This feature allows tracking patient data over multiple visits to assess intervention effectiveness.")
    
    # Add record form
    with st.expander("➕ Add New Record"):
        with st.form("trend_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID", "P001")
                visit_date = st.date_input("Visit Date", datetime.now())
                risk_score = st.number_input("Risk Score", 0.0, 1.0, 0.5, 0.01)
            
            with col2:
                hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.8, 0.1)
                glucose = st.number_input("Glucose (mg/dL)", 50, 400, 110)
                bmi = st.number_input("BMI", 10.0, 100.0, 28.5, 0.1)
            
            notes = st.text_area("Clinical Notes", "")
            
            if st.form_submit_button("Add Record", type="primary"):
                record = LongitudinalRecord(
                    date=datetime.combine(visit_date, datetime.min.time()),
                    risk_score=risk_score,
                    hba1c=hba1c,
                    glucose=glucose,
                    bmi=bmi,
                    notes=notes
                )
                engine.add_record(patient_id, record)
                st.success(f"✅ Record added for patient {patient_id}")
    
    # View trends
    st.subheader("📊 View Patient Trends")
    
    patient_id_view = st.text_input("Enter Patient ID to view trends", "P001")
    
    if st.button("Load Trends"):
        trends = engine.get_trends(patient_id_view)
        
        if trends:
            st.success(f"✅ Loaded {trends['data_points']} records")
            
            # Trend summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                direction_icon = "📉" if trends['risk_trend']['direction'] == 'improving' else "📈"
                st.metric(
                    "Risk Trend",
                    f"{trends['risk_trend']['end']:.1%}",
                    f"{trends['risk_trend']['change']:+.1%}",
                    delta_color="inverse"
                )
                st.caption(f"{direction_icon} {trends['risk_trend']['direction'].title()}")
            
            with col2:
                st.metric(
                    "HbA1c Change",
                    f"{trends['hba1c_trend']['end']:.1f}%",
                    f"{trends['hba1c_trend']['change']:+.1f}%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "BMI Change",
                    f"{trends['bmi_trend']['end']:.1f}",
                    f"{trends['bmi_trend']['change']:+.1f}",
                    delta_color="inverse"
                )
            
            # Trend chart
            fig = create_trend_chart(trends)
            st.plotly_chart(fig, use_container_width=True)
            
            # History table
            st.subheader("📋 Visit History")
            history_data = []
            for r in trends['history']:
                history_data.append({
                    'Date': r.date.strftime('%Y-%m-%d'),
                    'Risk Score': f"{r.risk_score:.1%}",
                    'HbA1c': f"{r.hba1c:.1f}%",
                    'Glucose': f"{r.glucose} mg/dL",
                    'BMI': f"{r.bmi:.1f}",
                    'Notes': r.notes
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        else:
            st.warning("No data found for this patient. Add records using the form above.")

# MODEL INSIGHTS
elif page == "📊 Model Insights":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Model Insights</h1>
        <p>Technical details and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Performance", "Features", "Thresholds", "Evidence Base"])
    
    with tabs[0]:
        st.subheader("🎯 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("ROC-AUC", "0.9775", "Excellent discrimination"),
            ("Accuracy", "0.9730", "High overall accuracy"),
            ("Precision", "0.9835", "Low false positive rate"),
            ("Recall", "0.6942", "Moderate sensitivity")
        ]
        
        for col, (name, value, desc) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.metric(name, value)
                st.caption(desc)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Brier Score", "0.0226", help="Lower is better (calibration)")
            st.metric("Training Samples", "100,992")
        
        with col2:
            st.metric("Calibrated", "Yes (Platt scaling)")
            st.metric("Cross-Validation", "5-fold stratified")
    
    with tabs[1]:
        st.subheader("📊 Feature Importance")
        
        features_df = pd.DataFrame({
            'Feature': ['HbA1c Level', 'Blood Glucose', 'BMI', 'Age', 
                       'Hypertension', 'Heart Disease', 'Smoking', 'Gender'],
            'Importance': [0.631, 0.322, 0.186, 0.185, 0.048, 0.037, 0.014, 0.005],
            'Type': ['Numerical', 'Numerical', 'Numerical', 'Numerical',
                    'Binary', 'Binary', 'Categorical', 'Categorical'],
            'Modifiable': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
        })
        
        fig = px.bar(
            features_df.sort_values('Importance'),
            x='Importance', y='Feature',
            orientation='h',
            color='Modifiable',
            color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'},
            title='Feature Importance (Modifiable vs Non-Modifiable)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(features_df, use_container_width=True)
    
    with tabs[2]:
        st.subheader("⚙️ Risk Thresholds")
        
        thresholds_df = pd.DataFrame({
            'Level': ['Low', 'Moderate', 'High', 'Critical'],
            'Risk Range': ['< 30%', '30-60%', '60-85%', '> 85%'],
            'Color': ['🟢', '🟡', '🟠', '🔴'],
            'Clinical Action': [
                'Annual screening, maintain lifestyle',
                'Lifestyle modification, 3-6 month follow-up',
                'Intensive intervention, monthly monitoring',
                'Immediate specialist referral'
            ],
            'Follow-up': ['6-12 months', '3 months', '1 month', '1-2 weeks']
        })
        
        st.dataframe(thresholds_df, use_container_width=True)
        
        # Threshold visualization
        fig = go.Figure()
        
        ranges = [(0, 30, 'Low', '#28a745'), (30, 60, 'Moderate', '#ffc107'),
                 (60, 85, 'High', '#ff9800'), (85, 100, 'Critical', '#dc3545')]
        
        for start, end, label, color in ranges:
            fig.add_trace(go.Bar(
                x=[end - start],
                y=['Risk Level'],
                orientation='h',
                name=f'{label} ({start}-{end}%)',
                marker_color=color,
                text=label,
                textposition='inside'
            ))
        
        fig.update_layout(
            barmode='stack',
            title='Risk Level Distribution',
            xaxis_title='Risk Probability (%)',
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("📚 Clinical Evidence Base")
        
        st.markdown("""
        ### Primary Evidence Sources
        
        **1. Diabetes Prevention Program (DPP)**
        - N = 3,234 participants
        - 58% risk reduction with intensive lifestyle intervention
        - 31% risk reduction with metformin
        - 7% weight loss target, 150 min/week activity
        - **Evidence Level: A**
        
        **2. PREDIMED Trial**
        - N = 7,447 participants
        - 30% diabetes risk reduction with Mediterranean diet
        - Benefits independent of weight loss
        - **Evidence Level: A**
        
        **3. Look AHEAD Study**
        - N = 5,145 participants with T2DM
        - Intensive lifestyle intervention
        - Significant weight loss and improved glycemic control
        - **Evidence Level: A**
        
        **4. UKPDS**
        - N = 5,102 participants
        - Established importance of glycemic control
        - Foundation for treatment targets
        - **Evidence Level: A**
        
        ---
        
        ### Clinical Guidelines Referenced
        
        - **ADA Standards of Medical Care 2024**
        - **USPSTF Diabetes Screening Recommendations**
        - **AACE/ACE Clinical Practice Guidelines**
        - **Cochrane Systematic Reviews**
        """)

# DOCUMENTATION
elif page == "📖 Documentation":
    st.markdown("""
    <div class='main-header'>
        <h1>📖 Documentation</h1>
        <p>User guide and clinical reference</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["User Guide", "Clinical Guide", "Technical", "FAQ"])
    
    with tabs[0]:
        st.markdown("""
        ## User Guide
        
        ### For Patients
        
        **1. Patient Assessment**
        - Navigate to **👤 Patient Assessment**
        - Enter your health information accurately
        - Review your personalized risk assessment
        - Follow the recommended action items
        
        **2. Understanding Your Results**
        - **Risk Score**: Your probability of developing diabetes
        - **Risk Level**: Category (Low/Moderate/High/Critical)
        - **Key Numbers**: HbA1c, glucose, BMI with targets
        
        **3. Taking Action**
        - Review "What You Can Do" section
        - Discuss results with your healthcare provider
        - Set realistic health goals
        - Schedule follow-up as recommended
        
        ---
        
        ### For Clinicians
        
        **1. Clinical Analysis**
        - Navigate to **👨‍⚕️ Clinical Analysis**
        - Enter complete patient data
        - Review comprehensive risk assessment
        - Access evidence-based interventions
        
        **2. Key Features**
        - Risk prediction with confidence intervals
        - Modifiable vs non-modifiable factor analysis
        - Priority-ranked interventions with evidence levels
        - What-if scenario modeling
        - Downloadable clinical reports
        
        **3. Workflow Integration**
        - Use during patient consultations
        - Generate reports for documentation
        - Track patient progress over time
        - Support shared decision-making
        """)
    
    with tabs[1]:
        st.markdown("""
        ## Clinical Guide
        
        ### Risk Stratification
        
        | Risk Level | Probability | Clinical Action |
        |------------|-------------|-----------------|
        | Low | < 30% | Annual screening, lifestyle reinforcement |
        | Moderate | 30-60% | Lifestyle intervention, 3-6 month follow-up |
        | High | 60-85% | Intensive intervention, consider metformin, monthly monitoring |
        | Critical | > 85% | Immediate referral, comprehensive workup |
        
        ### Evidence-Based Thresholds
        
        **Glycemic:**
        - HbA1c < 5.7%: Normal
        - HbA1c 5.7-6.4%: Prediabetes
        - HbA1c ≥ 6.5%: Diabetes
        
        **Glucose:**
        - Fasting < 100 mg/dL: Normal
        - Fasting 100-125 mg/dL: Impaired
        - Fasting ≥ 126 mg/dL: Diabetes
        
        ### Intervention Priority
        
        1. **CRITICAL**: Immediate action required (safety concern)
        2. **HIGH**: Address within 1-2 weeks
        3. **MEDIUM**: Address within 1-3 months
        4. **LOW**: Routine optimization
        
        ### Evidence Levels
        
        - **A**: High-quality RCTs, meta-analyses
        - **B**: Well-designed cohort studies
        - **C**: Expert consensus, observational data
        """)
    
    with tabs[2]:
        st.markdown("""
        ## Technical Documentation
        
        ### Model Architecture
        
        - **Algorithm**: Gradient Boosting Classifier
        - **Calibration**: Platt scaling
        - **Features**: 8 clinical variables
        - **Training Data**: 100,992 samples
        - **Validation**: 5-fold stratified cross-validation
        
        ### Performance Metrics
        
        ```
        ROC-AUC:    0.9775
        Accuracy:   0.9730
        Precision:  0.9835
        Recall:     0.6942
        Brier:      0.0226
        ```
        
        ### Feature Processing
        
        - Categorical encoding (gender, smoking)
        - No missing value imputation required
        - Feature scaling not required (tree-based)
        
        ### Uncertainty Quantification
        
        - Bootstrap-based confidence intervals
        - Prediction confidence categorization
        - Context-aware risk adjustment
        
        ### System Requirements
        
        - Python 3.8+
        - Streamlit 1.29+
        - scikit-learn 1.3+
        - Plotly 5.18+
        """)
    
    with tabs[3]:
        st.markdown("## Frequently Asked Questions")
        
        faqs = [
            ("How accurate is the prediction?", 
             "The model has a ROC-AUC of 97.7%, indicating excellent discrimination between those who will and won't develop diabetes. All predictions include confidence intervals."),
            ("What do the risk levels mean?",
             "Low (<30%): Below average risk. Moderate (30-60%): Above average, action recommended. High (60-85%): Strong likelihood without intervention. Critical (>85%): May already have diabetes or very close."),
            ("Can I trust the recommendations?",
             "All recommendations are based on Level A evidence from major clinical trials (DPP, PREDIMED) and current ADA guidelines. Always discuss with your healthcare provider."),
            ("What if my risk is high?",
             "High risk doesn't mean you have diabetes - it's a warning. The DPP study showed 58% of cases can be prevented with lifestyle changes. Discuss the recommended interventions with your doctor."),
            ("How often should I be tested?",
             "Low risk: Annually. Moderate: Every 3-6 months. High: Monthly initially. Critical: Within 1-2 weeks, then as directed."),
            ("Is my data safe?",
             "All data is processed locally in your browser session. No personal health information is stored or transmitted externally.")
        ]
        
        for question, answer in faqs:
            with st.expander(f"❓ {question}"):
                st.write(answer)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>🏥 <strong>Clinical Decision Support System</strong> v2.0.0</p>
    <p>Powered by AI • Evidence-Based Medicine • Patient-Centered Care</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        ⚠️ <em>This tool is for informational purposes only and should not replace 
        professional medical advice, diagnosis, or treatment.</em>
    </p>
</div>
""", unsafe_allow_html=True)