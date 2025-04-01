#!/bin/sh 

dataset="entity_questions"
linguistics=("readability" "back_translated" "edited_query_char" "formality" "politeness")
modifications=("modified" "original")

for linguistic in "${linguistics[@]}"; do
    for modified in "${modifications[@]}"; do
        python ../passage_retrieval.py \
            --dataset "$dataset" --linguistic "$linguistic" --modified "$modified" \
            --n_docs 100 \
            --lowercase --normalize_text \
            --save_or_load_index \
            --m 64 --nbits 8
    done
done