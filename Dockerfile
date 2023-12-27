FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50052

CMD ["python", "./server.py"]
