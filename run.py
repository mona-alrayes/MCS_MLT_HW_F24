#!/usr/bin/env python3
from app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem"
    )