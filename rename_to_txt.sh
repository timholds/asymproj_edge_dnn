#!/bin/bash

# Reformat edgelists with .txt extension to pass to asymproj to create training data for optimized node embeddings with Watch Your Step
# Run for default directory or pass a directory of edgelists as an argument

if [ -z "$1" ]; then
  echo Running for $DIR
  echo "To run for another directory, pass an argument with the directory that contains the edgelists for all years"
  DIR=/mnt/raid/data/features/top84/embeddings/edgelists
  FILES=$DIR/*
  for f in $FILES; do
    mv $f $f.txt
  done
else
  DIR=$1
  FILES=$1/edgelists/*
  for f in $FILES; do
    mv $f $f.txt
  done 
fi

