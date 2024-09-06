# main.py

import uvicorn
from src.app_module import http_server

if __name__ == "__main__":
    uvicorn.run(http_server, host="0.0.0.0", port=8000)
