import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin@loanapp.com")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "SecureAdminPass123!")

settings = Settings()