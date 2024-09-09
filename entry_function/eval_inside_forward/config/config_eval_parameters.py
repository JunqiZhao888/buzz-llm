import argparse


def config_junqi_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)

    parser.add_argument(
        "--split", type=str, default="validation", choices=["validation", "test"]
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=40,
    )

    ######################
    # full
    # streaming
    # local
    # buzz
    # buzz_fast
    # buzz_no_max
    # h2o
    parser.add_argument(
        "--my_eval_type",
        type=str,
        default="buzz"
    )
    parser.add_argument("--sample_threshold", type=int, default=400)
    parser.add_argument("--sample_stride", type=int, default=5)
    ######################

    parser.add_argument("--enable_start_recent_kv_cache", default=True)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=160)
    parser.add_argument("--enable_pos_shift", default=True)

    args = parser.parse_args()
    return args
