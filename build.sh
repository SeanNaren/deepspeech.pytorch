#!/bin/bash


WHOAMI=$(whoami)
CONTAINERNAME=deepspeech.pytorch
TAGNAME="$WHOAMI/$CONTAINERNAME"


##
## Then build the container
##
docker build --file="Dockerfile" --tag "$TAGNAME" .

if [[ $? -eq 0 ]] ; then
    echo "Successfully built new container, proceeding to test"
else
    echo "Failed to build container, skipping test."
    exit 1
fi


