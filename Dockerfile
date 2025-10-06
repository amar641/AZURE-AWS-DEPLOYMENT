FROM python:3.8-slim-buster 
WORKDIR /app
COPY ./app ./
COPY requirements.txt ./
RUN apt update -y && apt install awscli -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]