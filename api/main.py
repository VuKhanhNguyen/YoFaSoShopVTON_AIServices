"""Entry point Uvicorn. Khóa thật chỉ đọc từ .env/environment lúc khởi động."""
from api.application import create_app

app = create_app()
