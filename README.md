# BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for Efficient LLM Inference

## Abstract
Large language models (LLMs) have become essential in natural language processing, powering a wide range of applications. However, these models often struggle with inference speed and computational efficiency, which can hinder real-time deployment and degrade user experience. The key-value (KV) cache mechanism has been introduced to alleviate computational overhead during the prefilling and decoding stages of transformer models. While previous approaches, such as H2O, aim to retain the most important token information, they still face challenges in preserving contextual understanding. In this paper, we introduce BUZZ, a new KV caching algorithm designed to leverage structured contextual information, minimize cache memory usage and enhance inference speed. BUZZ utilizes a beehive-structured sparse cache, implementing a sliding window to capture the most recent information, segmenting historical tokens into chunks, and dynamically selecting the most important tokens within local neighborhoods. We evaluate BUZZ on four real-world datasets—CNN/Daily Mail, XSUM, Wikitext, and 10-QA. Our results show that BUZZ (1) reduces cache memory usage in LLM inference by **2.5×** while maintaining above 99% accuracy in long-text summarization, and (2) surpasses state-of-the-art performance in multi-document question answering by **7.69%** under the same cache memory limit, where full cache methods encounter out-of-memory issues. Additionally, we validate that BUZZ operates with log(n) time complexity, achieving significant inference speedup.

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
  