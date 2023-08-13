FROM python:3.9

COPY requirements.txt setup.py /workdir/
COPY app/ /workdir/app/
COPY models/ /workdir/models/
COPY src/ /workdir/src

WORKDIR /workdir

RUN pip install -r requirements.txt

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]