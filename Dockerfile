# Standard image includes required build tools
FROM python:3.12

WORKDIR /app

# Copy only requirements first
COPY ./requirements.txt /app/requirements.txt

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --upgrade pip --default-timeout=100 && \
    pip install --no-cache-dir --default-timeout=100 -r /app/requirements.txt

COPY ./src /app

# Start the app
CMD ["python", "main.py"]
