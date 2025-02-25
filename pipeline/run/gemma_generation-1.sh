#!/bin/bash

cd ..
pwd

dataset=("popqa" "entity_questions")
linguistics=("back_translated" "edited_query_char" "formality" "politeness")
modified=("original" "modified")

retrieval=("ModernBERT")

vllm_url="http://babel-0-37:8080/v1"
model_name="google/gemma-2-9b-it"

# Iterate over all combinations of arg1, arg2, and arg3
for arg1 in "${dataset[@]}"; do
    for arg2 in "${linguistics[@]}"; do
        for arg3 in "${modified[@]}"; do
            for arg4 in "${retrieval[@]}"; do
                # Run the Python script with the current combination of arguments
                echo "Generation, args: "$arg1", "$arg2", "$arg3", "$arg4", "$vllm_url", "$model_name""
                # skip if linguistic is back_translated and modified is original
                if [ "$arg2" == "back_translated" ] && [ "$arg3" == "original" ]; then
                    echo "skipping"
                    continue
                fi

                python a_vllm_generation.py \
                    --dataset "$arg1" --linguistics "$arg2" --modified "$arg3" \
                    --retrieval "$arg4" --n_docs 5 \
                    --vllm_url "$vllm_url" --model_name "$model_name"

                # echo "Evaluation, args: $arg1, $arg2, $arg3"
                # python c_eval_generation.py --dataset $arg1 --property $arg2 --modified $arg3 --result_path /data/user_data/tianyuca/QL_result/popularity_quantile
            done
        done
    done
done