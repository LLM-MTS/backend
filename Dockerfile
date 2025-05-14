FROM python:3.10-slim

# set your working directory to where main.py actually is
WORKDIR /app/backend

# copy only the backend requirements & install
COPY backend/requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# copy the backend code into /app/backend
COPY backend .

ENV PYTHONUNBUFFERED=1

# point uvicorn at the FastAPI instance in main.py (often named "app")
# make sure your main.py does: app = FastAPI()
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
