FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y gcc libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (use minimal dependencies!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only essential files
COPY main.py .

# Expose port and run
ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
