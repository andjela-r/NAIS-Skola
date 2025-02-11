FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/ || exit 1

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "python crud.py && uvicorn main:app --host 0.0.0.0 --port 8000"]