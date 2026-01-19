# Optimizing RAG Pipelines with RapidFire AI

This repository demonstrates how to build, evaluate, and optimize a **Retrieval-Augmented Generation (RAG)** pipeline for financial question answering using the [FiQA dataset](https://huggingface.co/datasets/explodinggradients/fiqa) and the **RapidFire AI** framework.

## Project Overview

The goal of this project is to move beyond hardcoded RAG configurations. By using RapidFire AI, we systematically test a "grid" of retrieval strategies to identify which settings provide the most accurate evidence for a language model to generate grounded financial advice.
We evaluate:

    2 Chunking Strategies: Comparing chunk_size of 256 vs. 128.

    2 Reranking Strategies: Comparing top_n values of 2 vs. 5.

    Total Combinations: 4 unique RAG configurations evaluated side-by-side.

## Pipeline Architecture

The notebook implements a full RAG workflow:

    Document Loading: Loads a financial corpus of documents and posts.

    Vector Storage: Uses FAISS with sentence-transformers/all-MiniLM-L6-v2 embeddings.

    Reranking: Employs a cross-encoder/ms-marco-MiniLM-L6-v2 to refine retrieved results.

    Generation: Utilizes a lightweight Qwen2.5-0.5B-Instruct model via vLLM for speed and efficiency.

    Evaluation: Computes real-time metrics including Precision, Recall, NDCG@5, and MRR.

## Customizing the Generator Model

The generator is defined using the RFvLLMModelConfig wrapper. This allows you to tune how the LLM processes the retrieved context and generates the final answer.
Key Configuration Knobs

In the vllm_config1 section of the notebook, you can modify the following parameters to suit your hardware or quality requirements:

    model: Swap "Qwen/Qwen2.5-0.5B-Instruct" for larger models (e.g., Llama-3) if your GPU memory allows.

    gpu_memory_utilization: Set to a decimal (e.g., 0.25) to limit how much VRAM vLLM reserves, which is helpful when running embeddings on the same card.

    max_model_len: Adjust the context window (e.g., 3000) to accommodate more retrieved chunks.

    sampling_params: Control creativity and length via temperature (higher for more variety) and max_tokens.

Python

vllm_config1 = RFvLLMModelConfig(
    model_config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 3000,
        # ... other hardware settings
    },
    sampling_params={
        "temperature": 0.8,
        "max_tokens": 128,
    },
    rag=rag_gpu, # Links the generator to your retrieval search space
)

## Key Metrics Tracked
RapidFire AI provides online aggregation of retrieval metrics, showing estimates and confidence intervals in real-time.

| Metric | Description |
| :--- | :--- |
| **NDCG@5** | **Normalized Discounted Cumulative Gain at rank 5**; rewards relevant documents found higher in the list. |
| **MRR** | **Mean Reciprocal Rank**; measures how quickly the first relevant document is found. |
| **Precision** | The fraction of retrieved documents that are actually relevant. |
| **Recall** | The fraction of all relevant documents that were successfully retrieved. |
| **F1 Score** | The harmonic mean of Precision and Recall for a balanced single score. |

---

## Getting Started

### Prerequisites
* A Python environment (3.10+) or Google Colab.
* A GPU is required for the vLLM generator and embedding steps.

### Installation
```bash
pip install rapidfireai
rapidfireai init --evals
