# Base image
FROM python:3.8-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran libopenblas-dev liblapack-dev awscli && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
