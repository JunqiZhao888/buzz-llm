import torch
# import sys
# sys.path.append("/content/buzz-llm")
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_streaming import StartRecentKVCache
from entry_function.deprecated.llm_cached.cache_entity.kv_cache_buzz_with_no_max import FixedStrideKVCache
from entry_function.deprecated.llm_cached.utils import config_eval, load
from draw_res_graph import cal_rouge_score


@torch.no_grad()
def my_inference_v2(model_m, tokenizer_t, prompt, cache=None, max_gen_len=1000, file_prefix="", task="one"):
    input_ids = tokenizer_t(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_m.device)
    outputs = model_m(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        output_attentions=True
    )
    past_key_values = outputs.past_key_values
    if cache is not None:
        past_key_values = cache(past_key_values, outputs.attentions)
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    res = ""
    pos = 0
    generated_text = None

    lens = open("cache_lens.txt", "a", encoding="UTF-8")

    for _ in range(max_gen_len):
        lens.write(str(past_key_values[0][0].size(2)) + '\n')
        outputs = model_m(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        if cache is not None:
            past_key_values = cache(past_key_values, outputs.attentions)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer_t.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        now = len(generated_text) - 1
        if now > pos:
            res += " ".join(generated_text[pos:now]) + " "
            pos = now
        if pred_token_idx == tokenizer_t.eos_token_id:
            break
    res += " ".join(generated_text[pos:]) + " "

    file_my_a = open(f"{file_prefix}_{task}_res.txt", "a", encoding="utf-8")
    file_my_a.write(res + '\n\n\n')
    return res


if __name__ == "__main__":

    device = "cuda"

    args = config_eval()
    model, tokenizer = load(args.model_name_or_path)
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

        print(f"run type {run_type}")

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

    file_q = open("one_long_q.txt", "r", encoding="utf-8")
    q = file_q.read()
    question = f"Simplify the article to 50 tokens: {q}"

    gen = my_inference_v2(model, tokenizer, question, cache=kv_cache, file_prefix=run_type)

    answer_f = open("one_long_a.txt")
    print(cal_rouge_score(gen, answer_f.read()))
