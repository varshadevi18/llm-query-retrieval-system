FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# Use prebuilt binaries to avoid compiling Rust/tokenizers
RUN pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
