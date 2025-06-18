from pydantic import BaseModel
from datetime import datetime

class ApplicationCreate(BaseModel):
    applicant_name: str
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: float
    property_area: str

class ApplicationUpdate(BaseModel):
    status: str

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ModelMetricsCreate(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rf_accuracy: float
    rf_precision: float
    rf_recall: float
    rf_f1: float