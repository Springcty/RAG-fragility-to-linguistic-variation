
# Out of Style: RAG's Fragility to Linguistic Variation

[Abstract](https://arxiv.org/abs/xxx): Despite the impressive performance of Retrieval-augmented Generation (RAG) systems across various NLP benchmarks, their robustness in handling real-world user-LLM interaction queries remains largely underexplored. This presents a critical gap for practical deployment, where user queries exhibit greater linguistic diversity and can trigger cascading errors across interdependent RAG components. In this work, we systematically analyze how varying four linguistic dimensions (formality, readability, politeness, and grammatical correctness) impact RAG performance. We evaluate two retrieval models and nine LLMs, ranging from 3 to 72 billion parameters, across four information-seeking Question Answering (QA) datasets. Our results reveal that linguistic reformulations significantly impact both retrieval and generation stages, leading to a performance drop of up to 40.41\% in Recall@5 scores for less formal queries and 38.86\% in answer match scores for queries containing grammatical errors. Notably, RAG systems exhibit greater sensitivity to such variations compared to LLM-only generations, highlighting their vulnerability to error propagation due to linguistic shifts. These findings highlight the need for improved robustness techniques to enhance reliability in diverse user interactions.

## 📂 Project Structure

- `LLM_generation/` – Contains the vllm inference code and generation stage evaluation code.
- `retrieval/` – Contains the retrieval stage implementation code based on retrieval model `Contriever` and `ModernBERT`.
- `query_rewriting/` – Contains the code for rewriting linguistically varied queries.

## 🚀 Running Experiments

Run rewriting experiments:

```bash
@Neel
```

Run retrieval experiment using `Contriever`:
```bash
@Neel
```

Run retrieval experiment using `ModernBERT`:

```bash
# Encode retrieval corpus via distributed training on clusters by SLURM
sbatch retrieval/ModernBERT/script/encode.sh

# Run retrieval experiment
bash retrieval/ModernBERT/script/retrieval.sh
```

Run LLM generation experiment using vllm:

```bash
# Start OpenAI-compatible server based on vllm
sbatch LLM_generation/script/vllm_load_model.sh

# Run vllm inference
bash LLM_generation/script/vllm_generation.sh

# Run generation results evaluation
bash LLM_generation/script/eval.sh
```

## 📜 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.


## 📖 Citation

If you use this work, please cite our paper:

```

```


