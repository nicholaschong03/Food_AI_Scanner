FROM python:3.12.9

WORKDIR /code

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]