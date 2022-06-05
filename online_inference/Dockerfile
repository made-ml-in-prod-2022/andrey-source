FROM python:3.8.10-slim
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt --no-cache-dir

COPY model ./model
COPY app.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="model/svm_model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
