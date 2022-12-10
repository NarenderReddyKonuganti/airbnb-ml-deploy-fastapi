# Get the Fast API image with Python version 3.7
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Create the directory for the container
WORKDIR /app
COPY requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py ./

# Copy the serialized model and the vectors
COPY ./models/LRM.sav ./models/LRM.sav

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]