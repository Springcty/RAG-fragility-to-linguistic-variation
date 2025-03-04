#!/bin/bash

cd ..
pwd

dataset=("popqa" "entity_questions" "ms_marco" "natural_questions")
linguistics=("readability" "back_translated" "edited_query_char" "formality" "politeness")
modified=("original" "modified")

retrieval=("ModernBERT")
# retrieval=("None")

vllm_url="http://babel-4-29:8092/v1"
model_name="Qwen/Qwen2.5-72B-Instruct"

# Iterate over all combinations of arg1, arg2, and arg3
for arg1 in "${dataset[@]}"; do
    for arg2 in "${linguistics[@]}"; do
        for arg3 in "${modified[@]}"; do
            for arg4 in "${retrieval[@]}"; do
                # Run the Python script with the current combination of arguments
                echo "Generation, args: "$arg1", "$arg2", "$arg3", "$arg4", "$vllm_url", "$model_name""
                
                # python a_vllm_generation.py \
                python b_vllm_generation_preempt.py \
                    --dataset "$arg1" --linguistics "$arg2" --modified "$arg3" \
                    --retrieval "$arg4" --n_docs 5 \
                    --vllm_url "$vllm_url" --model_name "$model_name"

                # python b_vllm_none_retrieval.py \
                #     --dataset "$arg1" --linguistics "$arg2" --modified "$arg3" \
                #     --vllm_url "$vllm_url" --model_name "$model_name"
            done
        done
    done
done