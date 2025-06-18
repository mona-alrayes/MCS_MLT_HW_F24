#!/bin/bash

# Install requirements without dependencies for large packages
pip install --no-cache-dir \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    supabase==2.3.0 \
    sqlalchemy==2.0.28 \
    psycopg2-binary==2.9.9 \
    python-dotenv==1.0.1 \
    joblib==1.3.2 \
    passlib[bcrypt]==1.7.4 \
    python-jose[cryptography]==3.3.0 \
    jinja2==3.1.3 \
    python-multipart==0.0.9 \
    requests==2.31.0

# Install ML packages with minimal dependencies
pip install --no-cache-dir --no-deps \
    scikit-learn==1.3.0 \
    xgboost==2.0.3 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    seaborn==0.13.2 \
    matplotlib==3.8.0

# Clean up to reduce size
find . -type d -name '__pycache__' -exec rm -rf {} +
find . -type f -name '*.pyc' -delete