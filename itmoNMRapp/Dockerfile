FROM python:3.9

WORKDIR /itmoNMRapp
COPY requirements.txt /itmoNMRapp
RUN pip install --upgrade pip -r requirements.txt
COPY . /itmoNMRapp
EXPOSE 5000