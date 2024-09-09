import sys
sys.path.append("/content/buzz-llm")
from datasets import load_dataset

from entry_function.eval_inside_forward.config.config_eval_parameters import config_junqi_eval
from entry_function.eval_inside_forward.config.run_greedy_inference import my_inference_v3

from entry_function.deprecated.llm_cached.utils import load
from entry_function.eval_inside_forward.cache.cache_entity_collection import *
from entry_function.eval_inside_forward.cache.cache_coordinator import CacheCoordinator
from entry_function.eval_inside_forward.config import modify_llama_forward

from entry_function.deprecated.eval_outside_forward.draw_res_graph import cal_rouge_score

if __name__ == "__main__":
    # dataset = load_dataset("abisee/cnn_dailymail",
    #                        "2.0.0",
    #                        split="validation",
    #                        cache_dir="D:/pcharm/data_cache"
    #                        )

    dataset = load_dataset("EdinburghNLP/xsum", 
                            split="validation"
                            )

    args = config_junqi_eval()
    model, tokenizer = load(args.model_name_or_path)

    device = "cuda"
    k_seq_dim = None
    v_seq_dim = None

    run_type = args.my_eval_type
    sink_len = args.start_size
    window_len = args.recent_size
    sample_threshold_len = args.sample_threshold
    sample_stride_len = args.sample_stride

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
            raise ValueError(f"Cannot deal with {model.config.model_type}")

        kv_cache = None

        if run_type == "full":
            pass
        elif run_type == "buzz_no_max":
            kv_cache = BuzzKVCacheNoMax(sink_size=sink_len,
                                        window_size=window_len,
                                        stride_size=sample_stride_len,
                                        sample_threshold=sample_threshold_len,
                                        k_seq_dim=k_seq_dim,
                                        v_seq_dim=v_seq_dim, )
        elif run_type == "buzz_fast":
            kv_cache = BuzzKVCacheFast(sink_size=sink_len,
                                       window_size=window_len,
                                       stride_size=sample_stride_len,
                                       sample_threshold=sample_threshold_len,
                                       k_seq_dim=k_seq_dim,
                                       v_seq_dim=v_seq_dim, )
        elif run_type == "streaming":
            kv_cache = StreamingKVCache(start_size=sink_len,
                                        recent_size=window_len,
                                        k_seq_dim=k_seq_dim,
                                        v_seq_dim=v_seq_dim, )
        elif run_type == "local":
            kv_cache = StreamingKVCache(start_size=0,
                                        recent_size=sink_len + window_len,
                                        k_seq_dim=k_seq_dim,
                                        v_seq_dim=v_seq_dim, )
        elif run_type == "h2o":
            kv_cache = H2OKVCache(hh_size=sample_threshold_len,
                                  recent_size=window_len)
        elif run_type == "buzz":
            kv_cache = BuzzKVCacheWithAccumulation(sink_size=sink_len,
                                                   window_size=window_len,
                                                   stride_size=sample_stride_len,
                                                   sample_threshold=sample_threshold_len,
                                                   k_seq_dim=k_seq_dim,
                                                   v_seq_dim=v_seq_dim, )
        else:
            raise ValueError("Unsupported run type")
        if run_type != "full":
          modify_llama_forward.GLOBAL_KV_CACHE = CacheCoordinator(model.config.num_hidden_layers, kv_cache)
          if "llama" in model.config.model_type:
              from entry_function.eval_inside_forward.config.modify_llama_forward import \
                  junqi_enable_llama_customized_forward

              junqi_enable_llama_customized_forward(model)
          else:
              raise ValueError(f"Cannot deal with {model.config.model_type}")

    file_rouge_l = open(f"{run_type}_multi_rouge_l.txt", "a", encoding="utf-8")
    file_rouge_1 = open(f"{run_type}_multi_rouge_1.txt", "a", encoding="utf-8")
    file_rouge_2 = open(f"{run_type}_multi_rouge_2.txt", "a", encoding="utf-8")

    # """cnn daily"""
    # for i in range(0, args.num_samples):
    #     print(f"word {i + 1}")
    #     q = dataset["article"][i]
    #     question = f"Text: {q}\n\nSUMMARIZE the previous text to 50 words!"
    #     if run_type != "full":
    #       res = my_inference_v3(model, tokenizer, question, file_prefix=run_type, task="multi")
    #     modify_llama_forward.GLOBAL_KV_CACHE.cleanup_cache()
    #     score_l, score_1, score_2 = cal_rouge_score(res, dataset["highlights"][i])
    #     file_rouge_l.write(str(score_l) + '\n')
    #     file_rouge_1.write(str(score_1) + '\n')
    #     file_rouge_2.write(str(score_2) + '\n')

    """XSUM"""
    for i in range(0, args.num_samples):
        print(f"word {i + 1}")
        q = dataset["document"][i]
        question = f"Text: {q}\n\nSUMMARIZE the previous text to 50 words!"
        if run_type != "full":
           res = my_inference_v3(model, tokenizer, question, file_prefix=run_type, task="multi")
        modify_llama_forward.GLOBAL_KV_CACHE.cleanup_cache()
        score_l, score_1, score_2 = cal_rouge_score(res, dataset["summary"][i])
        file_rouge_l.write(str(score_l) + '\n')
        file_rouge_1.write(str(score_1) + '\n')
        file_rouge_2.write(str(score_2) + '\n')