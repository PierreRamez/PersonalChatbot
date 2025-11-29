# use python 3.10
FROM python:3.10

# set working directory
WORKDIR /code

# cp requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# create the data directory and set permissions (imp. for writing feedback.jsonl)
RUN mkdir -p /code/data && chmod 777 /code/data

# Copy the application code
COPY ./app.py /code/app.py

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
