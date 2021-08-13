#!/bin/bash
for FILE in bbc/*/*.txt
do
  echo $FILE
  python src/preprocess.py -in $FILE -out ${FILE%.txt}.sents
done
