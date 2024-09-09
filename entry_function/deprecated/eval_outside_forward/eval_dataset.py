# import sys
# sys.path.append("/content/buzz-llm")
from datasets import load_dataset
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_streaming import StartRecentKVCache
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_buzz_with_no_max import FixedStrideKVCache
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_buzz_with_max import FixedStrideKVCacheWithMax
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_buzz_with_max_no_window import FixedStrideKVCacheWithMaxNoWindow
from entry_function.deprecated.llm_cached.utils import config_eval, load
from draw_res_graph import cal_rouge_score
from eval_one_sample import my_inference_v2


if __name__ == "__main__":
    dataset = load_dataset("abisee/cnn_dailymail",
                           "2.0.0",
                           split="validation",
                           cache_dir="D:/pcharm/data_cache"
                           )


    args = config_eval()
    model, tokenizer = load(args.model_name_or_path)

    device = "cuda"
    k_seq_dim = None
    v_seq_dim = None

    run_type = args.my_eval_type
    sink = args.start_size
    window = args.recent_size
    sample_threshold = args.sample_threshold
    sample_stride = args.sample_stride

    if args.enable_start_recent_kv_cache and args.enable_pos_shift:
        if "llama" in model.config.model_type or "gpt_neox" in model.config.model_type:
            k_seq_dim = v_seq_dim = 2
        elif "mpt" in model.config.model_type:
            v_seq_dim = 2
            k_seq_dim = 3
        elif "pythia" in model.config.model_type:
            k_seq_dim = v_seq_dim = 2
        elif "falcon" in model.config.model_type:
            v_seq_dim = 1
            k_seq_dim = 1
        else:
            raise ValueError(f"got {model.config.model_type}")

        if run_type == "full":
            kv_cache = None
        elif run_type == "streaming":
            kv_cache = StartRecentKVCache(
                start_size=sink,
                recent_size=window,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        elif run_type == "local":
            kv_cache = StartRecentKVCache(
                start_size=0,
                recent_size=sink + window,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        elif run_type == "buzz_no_max":
            kv_cache = FixedStrideKVCache(
                sink_size=sink,
                window_size=window,
                stride_size=sample_stride,
                sample_threshold=sample_threshold,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        elif run_type == "buzz":
            kv_cache = FixedStrideKVCacheWithMax(
                sink_size=sink,
                window_size=window,
                stride_size=sample_stride,
                sample_threshold=sample_threshold,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        elif run_type == "buzz_no_window":
            kv_cache = FixedStrideKVCacheWithMaxNoWindow(
                sink_size=sink,
                stride_size=sample_stride,
                sample_threshold=sample_threshold,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim
            )
        else:
            raise ValueError("Unsupported run type")

        if "llama" in model.config.model_type:
            from entry_function.deprecated.llm_cached.forward_config.modify_llama import enable_llama_pos_shift_attention

            enable_llama_pos_shift_attention(model)
        elif "falcon" in model.config.model_type:
            from entry_function.deprecated.llm_cached.forward_config.modify_falcon import (
                enable_falcon_pos_shift_attention,
            )

            enable_falcon_pos_shift_attention(model)
        elif "gpt_neox" in model.config.model_type:
            from entry_function.deprecated.llm_cached.forward_config.modify_gpt_neox import (
                enable_gpt_neox_pos_shift_attention,
            )

            enable_gpt_neox_pos_shift_attention(model)
        elif "mpt" in model.config.model_type:
            pass
        else:
            raise ValueError(f"got {model.config.model_type}")
    else:
        kv_cache = None

    file_rouge_l = open(f"{run_type}_multi_rouge_l.txt", "a", encoding="utf-8")
    file_rouge_1 = open(f"{run_type}_multi_rouge_1.txt", "a", encoding="utf-8")
    file_rouge_2 = open(f"{run_type}_multi_rouge_2.txt", "a", encoding="utf-8")

    """cnn daily"""
    for i in range(0, args.num_samples):
        print(f"word {i + 1}")
        q = dataset["article"][i]
        question = f"Summarize the text to 50 words: {q}\n\nSUMMARIZE"
        res = my_inference_v2(model, tokenizer, question, cache=kv_cache, file_prefix=run_type, task="multi")
        score_l, score_1, score_2 = cal_rouge_score(res, dataset["highlights"][i])
        file_rouge_l.write(str(score_l) + '\n')
        file_rouge_1.write(str(score_1) + '\n')
        file_rouge_2.write(str(score_2) + '\n')

    """long bench needs A100"""
    # for i in range(0, args.num_samples):
    #   print(f"word {i + 1}")
    #   q = dataset["input"][i]
    #   c = dataset["context"][i]
    #   question = f"Context: {c}\n\nQuestion: {q}\n\nAnswer:"
    #   res = my_inference_v2(model, tokenizer, question, cache=kv_cache, file_prefix=run_type, task="multi")

    """XSUM"""
    # for i in range(0, args.num_samples):
    #     print(f"word {i + 1}")
    #     q = dataset["document"][i]
    #     question = f"Summarize the text to 50 words: {q}\n\nSUMMARIZE"

    #     res = my_inference_v2(model, tokenizer, question, cache=kv_cache, file_prefix=run_type, task="multi")
    #     score_l, score_1, score_2 = cal_rouge_score(res, dataset["summary"][i])
    #     file_rouge_l.write(str(score_l) + '\n')
    #     file_rouge_1.write(str(score_1) + '\n')
    #     file_rouge_2.write(str(score_2) + '\n')
