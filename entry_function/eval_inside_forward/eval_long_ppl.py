import sys
sys.path.append("/content/buzz-llm")

from tqdm import tqdm

from datasets import load_dataset
from torch.nn import CrossEntropyLoss

from entry_function.eval_inside_forward.config import modify_llama_forward
from entry_function.eval_inside_forward.config.config_eval_parameters import config_junqi_eval

from entry_function.deprecated.llm_cached.utils import load
from entry_function.eval_inside_forward.cache.cache_entity_collection import *
from entry_function.eval_inside_forward.cache.cache_coordinator import CacheCoordinator


args = config_junqi_eval()
model, tokenizer = load(args.model_name_or_path)
data = load_dataset("wikitext","wikitext-2-raw-v1",split="validation")
device = "cuda"
nlls = []
lens = []

loss_fn = CrossEntropyLoss(reduction="none")
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

f_nll = open(f"{run_type}_nll.txt", "a", encoding="utf-8")
f_len = open(f"{run_type}_len.txt", "a", encoding="utf-8")


def eval_with_no_buffer_for_each_word():
    modify_llama_forward.GLOBAL_EVICT_FLAG = True
    past_key_values = None
    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")
        #input_ids = tokenizer(text, return_tensors="pt").input_ids
        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))
        print(text)
        for idx in pbar:
            input_ids = encodings.input_ids[:, idx: idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1: idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            #print(past_key_values[0][0].size(k_seq_dim), file=f_len, flush=True)
            print(neg_log_likelihood.item(), file=f_nll, flush=True)
            num_eval_tokens += 1
            modify_llama_forward.GLOBAL_KV_CACHE.cleanup_layers()
        #     if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        #         break
        # if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        #     break


# def eval_with_buffer_for_each_word():
#     past_key_values = None
#     num_eval_tokens = 0
#     for text in data["text"][: args.num_samples]:
#         encodings = tokenizer(text, return_tensors="pt")

#         print(encodings.input_ids[:, :10])

#         seq_len = encodings.input_ids.size(1)
#         print(f"seq_len: {seq_len}")
#         pbar = tqdm(range(0, seq_len - 1))

#         for idx in pbar:
#             input_ids = encodings.input_ids[:, idx: idx + 1].to(device)
#             with torch.no_grad():
#                 outputs = model(
#                     input_ids,
#                     past_key_values=past_key_values,
#                     use_cache=True,
#                 )
#                 logits = outputs.logits.view(-1, model.config.vocab_size)
#                 past_key_values = outputs.past_key_values
#                 label = encodings.input_ids[:, idx + 1: idx + 2].to(logits.device).view(-1)
#                 neg_log_likelihood = loss_fn(logits, label)
#             nlls.append(neg_log_likelihood)
#             pbar.set_description(
#                 f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
#             )
#             print(past_key_values[0][0].size(k_seq_dim), file=f_len, flush=True)
#             print(neg_log_likelihood.item(), file=f_nll, flush=True)
#             num_eval_tokens += 1
#             if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
#                 break
#         if kv_cache is not None:
#             if past_key_values is not None:
#                 past_key_values = kv_cache(past_key_values)
#         if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
#             break

eval_with_no_buffer_for_each_word()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{run_type}_ppl.txt", "a",encoding="utf-8") as f_ppl:
    f_ppl.write(f"{ppl.item()}\n")
f_nll.close()
f_len.close()
f_ppl.close()