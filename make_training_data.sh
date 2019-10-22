#!/bin/bash
# First argument is the directory containing edgelists with txt extensions
if [ -z "$1" ]; then # no argumment passed, run for default directory
  DIR=/mnt/raid/data/features/top84/embeddings
  FILES=$DIR/edgelists/*.txt
  echo Creating training data from $FILES
  echo "To run for another directory, pass an argument with the directory that contains the edgelists for all years"

  for f in $FILES; do
    YEAR=${f: -17:-13}
    echo $YEAR
    echo $f/$YEAR
    python3 create_numpy_arrays.py --numum_walks=0 --input=$f --output_dir=/mnt/raid/data/features/top84/embeddings/optimized/training_data/$YEAR
  done
else
  DIR=$1
  FILES=$DIR/edgelists/*.txt
  for f in $FILES; do
    YEAR=${f: -17:-13}
    echo Creating training data from $f for $YEAR
    python3 create_numpy_arrays.py --numum_walks=0 --input=$f --output_dir=$DIR/optimized/training_data/$YEAR
  done
  echo Wrote all training data to $DIR/optimized/training_data/ 

fi
