#!/bin/bash

declare -a chapters = ("chapter01", "chapter02", "chapter02-end", "chapter03", "chapter03-end", "chapter04", "chapter04-end", "chapter05", "chapter05-end", "chapter06", "chapter06-end", "chapter07", "chapter07-end", "chapter08", "chapter08-end", "chapter09", "chapter09-end", "chapter10", "chapter10-end", "chapter11", "chapter11-end", "chapter12", "chapter12-end", "chapter13", "chapter13-end", "database_encryption")

mkdir /src/

for chpt in "${chapters[@]}"
do
    echo "Downloading $chapters"
    curl -L -O "/src/$chapter.zip" "https://github.com/NeilMadden/apisecurityinaction/archive/refs/heads/$chapter.zip"
    unzip "$chapter.zip" -d /src/
    rm "/src/$chapter.zip"
done
