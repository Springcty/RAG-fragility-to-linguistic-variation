#!/bin/sh 

dataset="popqa"
linguistics=("readability" "formality" "back_translated" "edited_query_char" "politeness")
modifications=("original" "modified")

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        python ../passage_retrieval_HyDE.py \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified" \
            --n_docs 100 \
            --lowercase --normalize_text \
            --save_or_load_index \
            --m 64 --nbits 8
    done
done