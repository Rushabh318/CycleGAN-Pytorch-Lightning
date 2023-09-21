#FROM continuumio/miniconda3
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime as base

RUN apt-get update && apt-get install nano
#RUN apt-get install nano

RUN mkdir /code
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt 
RUN pip install -r ./requirements.txt

COPY code/ /code
COPY conf.yml /code
COPY conf_cyclegan.yml /code
