FROM python:3.11-slim

# Java χρειάζεται για το Spark
RUN apt-get update && apt-get install -y \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "A_part.py", " B_part.py"]