FROM python:3.10-slim

WORKDIR /app

# Instalar herramientas del sistema y dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar pip packages (torch CPU y dependencias)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cpu \
        torchvision==0.16.0+cpu \
        torchaudio==2.1.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
