FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies for PDF processing and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Create virtual environment and install dependencies
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Install additional Python dependencies that might be needed
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers sentence-transformers pymupdf nltk

# Set the default command (can be overridden when running the container)
CMD ["python", "main.py"]