FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    adb \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytesseract Pillow

COPY automator.py /automator.py

CMD ["python", "-u", "/automator.py"]
