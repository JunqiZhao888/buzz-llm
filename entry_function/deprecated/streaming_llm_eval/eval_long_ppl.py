import torch
from tqdm import tqdm
import os
import sys
sys.path.append("/content/buzz-llm")
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_buzz_with_no_max import FixedStrideKVCache
from entry_function.deprecated.llm_cached.utils import config_eval, load


device = "cuda"

args = config_eval()

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
lens = []

loss_fn = CrossEntropyLoss(reduction="none")
k_seq_dim = None
v_seq_dim = None

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
    # kv_cache = StartRecentKVCache(
    #     start_size=1,
    #     recent_size=255,
    #     k_seq_dim=k_seq_dim,
    #     v_seq_dim=v_seq_dim,
    # )

    kv_cache = FixedStrideKVCache(
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

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

os.makedirs(args.output_dir, exist_ok=True)
f_nll = open(f"{args.output_dir}/nll.txt", "w")
f_len = open(f"{args.output_dir}/len.txt", "w")


def eval_with_no_buffer_for_each_word():
    past_key_values = None
    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

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
                if kv_cache is not None:
                    if past_key_values is not None:
                        past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(past_key_values[0][0].size(k_seq_dim), file=f_len, flush=True)
            print(neg_log_likelihood.item(), file=f_nll, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break


def eval_with_buffer_for_each_word():
    past_key_values = None
    num_eval_tokens = 0
    for text in data["text"][: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        print(encodings.input_ids[:, :10])

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

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
            print(past_key_values[0][0].size(k_seq_dim), file=f_len, flush=True)
            print(neg_log_likelihood.item(), file=f_nll, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        if kv_cache is not None:
            if past_key_values is not None:
                past_key_values = kv_cache(past_key_values)
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break


eval_with_no_buffer_for_each_word()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f_ppl:
    f_ppl.write(f"{ppl.item()}\n")

f_nll.close()
f_len.close()
f_ppl.close()