import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from .config import settings

# Supabase database configuration
DB_HOST = f"db.{settings.SUPABASE_URL.split('//')[1].split('.')[0]}.supabase.co"
ENCODED_PASSWORD = quote_plus(settings.SUPABASE_PASSWORD)
DATABASE_URL = f"postgresql://postgres:{ENCODED_PASSWORD}@{DB_HOST}:5432/postgres?sslmode=require"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def create_tables():
    Base.metadata.create_all(bind=engine)