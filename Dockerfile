FROM python:3.11-slim

COPY requirements-app.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ..
WORKDIR /code

#create cache directory and untar data there
RUN mkdir -p data/cache \
   && for f in data/deploy/*.tar; do tar -xf "$f" -C data/cache; done \
   && rm -rf data/deploy

EXPOSE 7860

CMD ["gunicorn", "app:server", "--chdir", "app", "-b", "0.0.0.0:7860", "--timeout", "180", "--workers", "1", "--threads", "4"]