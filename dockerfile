
FROM python:3.8-slim-buster AS builder

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# This stage installs Tesseract-OCR (might be incompatible with pip)
RUN apt-get update && apt-get install -y tesseract-ocr

FROM python:3.8-slim

WORKDIR /app

# Copy application files and libraries (including Tesseract-OCR)
COPY --from=builder /app .

COPY . .

CMD ["streamlit", "run", "app.py"]