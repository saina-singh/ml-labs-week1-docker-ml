# 1. Official lightweight Python image
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy dependencies and install without caching wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy training dataset and Python ML application
COPY data.csv .
COPY train_and_predict.py .

# 5. Create directory for output artifacts
RUN mkdir -p /app/output

# 6. Default execution: run the ML training & inference pipeline
CMD ["python", "train_and_predict.py"]