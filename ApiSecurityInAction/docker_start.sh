#!/bin/bash

if [[ -z "${CHAPTER}" ]]
then
    echo "CHAPTER env var not set, defaulting to last chapter."
    CHAPTER="chapter13-end"
fi

echo "Starting $CHAPTER"

cd "apisecurityinaction-$CHAPTER/natter-api"

mvn clean compile exec:java
