FROM python:3.7.7-slim-buster

RUN echo Updating existing packages, installing and upgrading python and pip.
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install --upgrade pip

RUN echo Copying Flask service into a service directory.
COPY ./flask_app /DisasterService
WORKDIR /DisasterService/app

# Note: requirements.txt was created automatically using 
# pip freeze > requirements.txt 
# from inside the conda enviroment
# There may be unused libraries here
RUN echo Installing Python packages listed in requirements.txt
RUN pip install --no-cache-dir -r ../requirements.txt

RUN echo Downloading nltk pack...
RUN python -m nltk.downloader punkt stopwords wordnet

RUN echo Starting python and starting the Flask service...

EXPOSE 3001
ENTRYPOINT ["python"]
CMD ["run.py"]