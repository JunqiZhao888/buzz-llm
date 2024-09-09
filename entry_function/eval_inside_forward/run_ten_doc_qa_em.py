import sys
sys.path.append("/content/buzz-llm")
import json

from entry_function.eval_inside_forward.config.config_eval_parameters import config_junqi_eval
from entry_function.eval_inside_forward.config.run_greedy_inference import my_inference_v3
from entry_function.deprecated.llm_cached.utils import load
from entry_function.eval_inside_forward.cache.cache_entity_collection import *
from entry_function.eval_inside_forward.cache.cache_coordinator import CacheCoordinator
from entry_function.eval_inside_forward.config import modify_llama_forward


def calc_em_score(predicted, actual):
	return 1 if predicted.strip().lower() == actual.strip().lower() else 0

def transform_to_prompt(jsonl_input):
	prompts = []
	for line in jsonl_input:
		data =json.loads(line)
		question = data["question"]
		ans = data["answers"][0]
		ctxs = data["ctxs"]
		formal = ""
		for ctx in ctxs:
			formal = formal + ctx["title"] + '\n' + ctx['text'] + '\n'
		prompt = f"Documents are following: {formal}. Based on the provided documents, give an brief answer to this question: {question}. The answer has to directly answer the question and be BRIEF. "
		prompts.append({"prompt" : prompt, "answer" : ans})
	return prompts

def read_jsonl_file(file_path):
	with open(file_path, 'r', encoding = 'utf-8') as file:
		return file.readlines()


if __name__ == "__main__":

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
      modify_llama_forward.GLOBAL_KV_CACHE = CacheCoordinator(model.config.num_hidden_layers, kv_cache)

      if "llama" in model.config.model_type:
          from entry_function.eval_inside_forward.config.modify_llama_forward import \
              junqi_enable_llama_customized_forward

          junqi_enable_llama_customized_forward(model)
      else:
          raise ValueError(f"Cannot deal with {model.config.model_type}")

  #file_em = open(f"{run_type}_10_em.txt", "a", encoding="utf-8")


  jsonl_file_path = "/content/nq-open-10_total_documents_gold_at_4.jsonl"
  jsonl_input = read_jsonl_file(jsonl_file_path)
  prompts = transform_to_prompt(jsonl_input)
  for i in range(0, args.num_samples):
    print(f"question {i + 1}")
    res = my_inference_v3(model, tokenizer, prompts[i]['prompt'], file_prefix=run_type, task="10")
    modify_llama_forward.GLOBAL_KV_CACHE.cleanup_cache()
    #em_score = calc_em_score(res, prompts[i]['answer'])
    #file_em.write(str(em_score) + '\n')
