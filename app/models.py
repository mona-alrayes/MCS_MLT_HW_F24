from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from .database import Base
from datetime import datetime

class LoanApplication(Base):
    __tablename__ = "loan_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    applicant_name = Column(String, index=True)
    gender = Column(String)
    married = Column(String)
    dependents = Column(String)
    education = Column(String)
    self_employed = Column(String)
    applicant_income = Column(Float)
    coapplicant_income = Column(Float)
    loan_amount = Column(Float)
    loan_amount_term = Column(Float)
    credit_history = Column(Float)
    property_area = Column(String)
    prediction = Column(String)
    probability = Column(Float)
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    rf_accuracy = Column(Float)
    rf_precision = Column(Float)
    rf_recall = Column(Float)
    rf_f1 = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)