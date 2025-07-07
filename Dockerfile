FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg git
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY app/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]