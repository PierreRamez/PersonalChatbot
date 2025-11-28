# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create the data directory and set permissions (Crucial for writing feedback.jsonl)
RUN mkdir -p /code/data && chmod 777 /code/data

# Copy the application code
# FIX: We copy 'app.py' specifically, correcting your build error
COPY ./app.py /code/app.py

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
