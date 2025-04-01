#!/bin/sh 

dataset="popqa"
linguistics=("readability" "politeness" "back_translated" "edited_query_char" "formality")
modifications=("modified" "original")

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        python ../passage_reranking.py \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified"
    done
done