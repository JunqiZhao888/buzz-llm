# BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inference

## Abstract
Large language models (LLMs) are essential in natural language processing but often struggle with inference speed and computational efficiency, limiting real-time deployment. The key-value (KV) cache mechanism reduces computational overhead in transformer models, but challenges in maintaining contextual understanding remain. 

In this paper, we propose **BUZZ**, a novel KV caching algorithm that leverages structured contextual information to minimize cache memory usage while enhancing inference speed. BUZZ employs a beehive-structured sparse cache, incorporating a sliding window to capture recent information and dynamically segmenting historical tokens into chunks to prioritize important tokens in local neighborhoods.

We evaluate BUZZ on four real-world datasets: **CNN/Daily Mail**, **XSUM**, **Wikitext**, and **10-QA**. Our results demonstrate that BUZZ:

1. Reduces cache memory usage by **2.5×** in LLM inference while maintaining over 99% accuracy in long-text summarization.
2. Surpasses state-of-the-art performance in multi-document question answering by **7.69%** under the same memory limit, where full cache methods encounter out-of-memory issues.
3. Demonstrates lower perplexity than state-of-the-art.

## Usage
### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
pip install rouge
pip install nltk
```

### News
- Our article is publishing in progress

### Run Buzz

```bash
CUDA_VISIBLE_DEVICES=0 python entry_function/eval_inside_forward/main_run.py
```

### Overview of Code
- entry_function/eval_inside_forward/cache contains cache structure
- entry_function/eval_inside_forward/config is used for 
  - config parameters, such as model name, KV Cache type, etc.
  - modify LLMs' forward function so that they can use our cache system
- Use entry_function/eval_inside_forward/main_run.py to start
  