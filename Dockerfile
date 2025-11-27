FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    wget gnupg unzip \
    chromium \
    chromium-driver \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY .github .github

EXPOSE 7860

CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]

