FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    adb \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytesseract Pillow numpy opencv-python-headless requests

COPY automator.py gem.py /

CMD ["python", "-u", "/automator.py"]
