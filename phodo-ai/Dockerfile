FROM python:3.9

RUN mkdir /app

WORKDIR /app

# RUN pip install -r requirements.txt

COPY . /app

CMD python manage.py runserver 0.0.0.0:8000