# QueryLinguistic

For Query Rewriting and the Retry Mechanism:
The rewrite pipeline with retries is in final_script_5000_copy.sh, which runs non_candidates.py
It basically checks if the exisiting files I had made with 10k examples have the top 5k we filtered for, and for all in the top5k that either don’t meet the criteria or don’t exist in the 10k, we rerun with 5 retries.
The initial 10k was made with test_rewrite_prompts.py
The prompts are in prompts.py
