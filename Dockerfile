FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model to avoid runtime issues
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]