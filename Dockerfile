FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper.py .
COPY property_analysis.py .
COPY main.py .

# Expose debug port
EXPOSE 5678

# Run the script
CMD ["python", "main.py"]