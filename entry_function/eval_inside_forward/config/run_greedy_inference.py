import torch
from entry_function.eval_inside_forward.config import modify_llama_forward


@torch.no_grad()
def my_inference_v3(model_m, tokenizer_t, prompt, max_gen_len=1000, file_prefix="", task="one"):
    # 问题阶段不evict，这是可以调整了，为了和以前保持一致
    modify_llama_forward.GLOBAL_EVICT_FLAG = True
    input_ids = tokenizer_t(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model_m.device)
    outputs = model_m(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
    )
    modify_llama_forward.GLOBAL_KV_CACHE.cleanup_layers()
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    res = ""
    pos = 0
    generated_text = None
    # 问题问完了，要开始evict了
    modify_llama_forward.GLOBAL_EVICT_FLAG = True
    lens = open("cache_lens.txt", "a", encoding="UTF-8")
    for _ in range(max_gen_len):
        lens.write(str(past_key_values[0][0].size(2)) + '\n')
        outputs = model_m(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        modify_llama_forward.GLOBAL_KV_CACHE.cleanup_layers()
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
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
