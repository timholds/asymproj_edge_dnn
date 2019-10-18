#!/bin/bash
DIR = /mnt/raid/data/embeddings/top84-12d/
FILES=/mnt/raid/data/embeddings/top84-12d/quanta.*.edgelist.txt

for f in $FILES; do
  YEAR=${f: -17:-13}
  echo $YEAR
  echo /mnt/raid/data/embeddings/top84-12d/optimized/$YEAR
  python3 create_numpy_arrays.py --numum_walks=0 --input=$f --output_dir=/mnt/raid/data/embeddings/top84-12d/optimized/$YEAR
done
