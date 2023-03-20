#!/bin/bash

docker build -f Dockerfile -t surprise-adapt:latest .
docker tag surprise-adapt:latest $USER/surprise-adapt:latest
docker push $USER/surprise_adapt:latest

