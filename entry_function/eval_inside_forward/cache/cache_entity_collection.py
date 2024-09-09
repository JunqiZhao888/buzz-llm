from entry_function.deprecated.llm_cached.cache_data_access.kv_crud_manager import *
import random

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

DIM_TO_SLICE_WITH_LOCAL_MAX = {
    1: slice1d_with_local_max,
    2: slice2d_with_local_max,
    3: slice3d_with_local_max,
}


class StreamingKVCache:
    def __init__(
            self,
            start_size=4,
            recent_size=300,
            k_seq_dim=2,
            v_seq_dim=2,
    ):
        print(f"StreamingKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values, attn_scores):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        k = past_key_values[0]
        v = past_key_values[1]
        return [
            torch.cat(
                [
                    self.k_slice(k, 0, self.start_size),
                    self.k_slice(k, seq_len - self.recent_size, seq_len),
                ],
                dim=self.k_seq_dim,
            ),
            torch.cat(
                [
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(v, seq_len - self.recent_size, seq_len),
                ],
                dim=self.v_seq_dim,
            )
        ]

    def __deepcopy__(self, meme):
        return StreamingKVCache(self.start_size, self.recent_size, self.k_seq_dim, self.v_seq_dim)

    def clear(self):
        # has nothing to clear
        pass


class BuzzKVCacheNoMax:
    def __init__(
            self,
            sink_size=4,
            window_size=180,
            stride_size=4,
            k_seq_dim=2,
            v_seq_dim=2,
            sample_threshold=180
    ):
        print(
            f"BuzzKVCache with no local max, sink size, stride size, sample threshold, window size: "
            f"{sink_size}, "
            f"{stride_size}, "
            f"{sample_threshold}, "
            f"{window_size}")
        self.sink_size = sink_size
        self.window_size = window_size
        self.stride_size = stride_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.sample_threshold = sample_threshold
        self.cur_sample_size = 0

    def get_sink_other_window(self, new_coming_words):
        new_coming_size = new_coming_words[0].size(self.k_seq_dim)
        if new_coming_size <= self.sink_size:
            return (new_coming_words,
                    [],
                    [])
        elif new_coming_size <= self.sink_size + self.window_size:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [

                ], [

                    self.k_slice(k, self.sink_size),
                    self.v_slice(v, self.sink_size),
                ])
        else:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [
                    self.k_slice(k, self.sink_size, -self.window_size),
                    self.v_slice(v, self.sink_size, -self.window_size),
                ], [
                    self.k_slice(k, -self.window_size),
                    self.v_slice(v, -self.window_size),
                ])

    def concat_three_parts(self, one, two, three):

        concat = []

        part_one_0 = one[0] if len(one) > 0 else None
        part_two_0 = two[0] if len(two) > 0 else None
        part_three_0 = three[0] if len(three) > 0 else None

        parts_0 = [p for p in [part_one_0, part_two_0, part_three_0] if p is not None]
        concat_0 = torch.cat(parts_0, dim=self.k_seq_dim) if len(parts_0) else None

        part_one_1 = one[1] if len(one) > 1 else None
        part_two_1 = two[1] if len(two) > 1 else None
        part_three_1 = three[1] if len(three) > 1 else None

        parts_1 = [p for p in [part_one_1, part_two_1, part_three_1] if p is not None]
        concat_1 = torch.cat(parts_1, dim=self.k_seq_dim) if len(parts_1) > 0 else None

        if concat_0 is not None:
            concat.append(concat_0)
        if concat_1 is not None:
            concat.append(concat_1)
        return concat

    def sample_extract(self, evicted, stride):
        k = evicted[0]
        v = evicted[1]
        return [
            self.k_slice(k, start=0, step=stride),
            self.v_slice(v, start=0, step=stride),
        ]

    def __call__(self, past_key_values, attn_scores):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.window_size + self.sink_size:
            return past_key_values
        sink, other, window = self.get_sink_other_window(past_key_values)
        other_size = other[0].size(self.k_seq_dim)
        if other_size <= self.sample_threshold:
            return past_key_values
        old_sample = [
            self.k_slice(other[0], 0, self.cur_sample_size),
            self.v_slice(other[1], 0, self.cur_sample_size)]
        smaller_stride = (self.stride_size + 1) // 2
        old_sample = self.sample_extract(old_sample, smaller_stride)
        evicted = [
            self.k_slice(other[0], self.cur_sample_size),
            self.v_slice(other[1], self.cur_sample_size)
        ]
        new_sample = self.sample_extract(evicted, self.stride_size)
        cur_sample = self.concat_three_parts(old_sample, new_sample, [])
        self.cur_sample_size = cur_sample[0].size(self.k_seq_dim)
        return self.concat_three_parts(sink, cur_sample, window)

    def __deepcopy__(self, meme):
        return BuzzKVCacheNoMax(self.sink_size,
                                self.window_size,
                                self.stride_size,
                                self.k_seq_dim,
                                self.v_seq_dim,
                                self.sample_threshold)

    def clear(self):
        self.cur_sample_size = 0


class BuzzKVCacheFast:
    def __init__(
            self,
            sink_size=4,
            window_size=4,
            stride_size=3,
            k_seq_dim=2,
            v_seq_dim=2,
            sample_threshold=180,
    ):
        print(
            f"BuzzKVCache with local max light weight, sink size, stride size, sample threshold, window size: "
            f"{sink_size}, "
            f"{stride_size}, "
            f"{sample_threshold}, "
            f"{window_size}")
        self.sink_size = sink_size
        self.window_size = window_size
        self.stride_size = stride_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.k_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[k_seq_dim]
        self.v_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[v_seq_dim]
        self.sample_threshold = sample_threshold
        self.cur_sample_size = 0

    def get_sink_other_window(self, new_coming_words):
        new_coming_size = new_coming_words[0].size(self.k_seq_dim)
        if new_coming_size <= self.sink_size:
            return (new_coming_words,
                    [],
                    [])
        elif new_coming_size <= self.sink_size + self.window_size:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [

                ], [

                    self.k_slice(k, self.sink_size),
                    self.v_slice(v, self.sink_size),
                ])
        else:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [
                    self.k_slice(k, self.sink_size, -self.window_size),
                    self.v_slice(v, self.sink_size, -self.window_size),
                ], [
                    self.k_slice(k, -self.window_size),
                    self.v_slice(v, -self.window_size),
                ])

    def concat_three_parts(self, one, two, three):

        concat = []

        part_one_0 = one[0] if len(one) > 0 else None
        part_two_0 = two[0] if len(two) > 0 else None
        part_three_0 = three[0] if len(three) > 0 else None

        parts_0 = [p for p in [part_one_0, part_two_0, part_three_0] if p is not None]
        concat_0 = torch.cat(parts_0, dim=self.k_seq_dim) if len(parts_0) > 0 else None

        part_one_1 = one[1] if len(one) > 1 else None
        part_two_1 = two[1] if len(two) > 1 else None
        part_three_1 = three[1] if len(three) > 1 else None

        parts_1 = [p for p in [part_one_1, part_two_1, part_three_1] if p is not None]
        concat_1 = torch.cat(parts_1, dim=self.k_seq_dim) if len(parts_1) > 0 else None

        if concat_0 is not None:
            concat.append(concat_0)
        if concat_1 is not None:
            concat.append(concat_1)
        return concat

    def sample_extract(self, evicted, stride):
        k = evicted[0]
        v = evicted[1]
        return [
            self.k_slice(k, start=0, step=stride),
            self.v_slice(v, start=0, step=stride),
        ]

    @staticmethod
    def de_repeat_q(hidden_states, n_de_rep):
        batch, num_of_q_heads, s_len, head_dim = hidden_states.shape
        if num_of_q_heads % n_de_rep != 0:
            raise ValueError(f"Cannot chunk because {num_of_q_heads} % {n_de_rep}")
        chunks = torch.chunk(hidden_states, chunks=n_de_rep, dim=1)
        return torch.sum(torch.stack(chunks, dim=0), dim=0)

    def deal_new_sample_with_local_max(self, k, v, attn_score):
        attn_score = self.de_repeat_q(attn_score, 8)
        # hh_score是一个头 * token len的矩阵，代表了在每个头下，每个token收获的注意力
        hh_score = attn_score.sum(0).sum(1)
        # 因为local max抽样的部分只有new evicted，要去掉sink, old sample, window
        hh_score = hh_score[:, self.sink_size + self.cur_sample_size:-self.window_size]
        return [self.k_slice_with_max(x=k, start=0, step=self.stride_size, attn_scores=hh_score, probability=1),
                self.v_slice_with_max(x=v, start=0, step=self.stride_size, attn_scores=hh_score, probability=1)]

    def __call__(self, past_key_values, attn_scores):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.window_size + self.sink_size:
            return past_key_values
        sink, other, window = self.get_sink_other_window(past_key_values)
        other_size = other[0].size(self.k_seq_dim)
        if other_size <= self.sample_threshold:
            return past_key_values
        old_sample = [
            self.k_slice(other[0], 0, self.cur_sample_size),
            self.v_slice(other[1], 0, self.cur_sample_size)]
        smaller_stride = (self.stride_size + 1) // 2
        old_sample = self.sample_extract(old_sample, smaller_stride)
        evicted = [
            self.k_slice(other[0], self.cur_sample_size),
            self.v_slice(other[1], self.cur_sample_size)
        ]
        new_sample = self.deal_new_sample_with_local_max(evicted[0], evicted[1], attn_scores)
        cur_sample = self.concat_three_parts(old_sample, new_sample, [])
        self.cur_sample_size = cur_sample[0].size(self.k_seq_dim)
        return self.concat_three_parts(sink, cur_sample, window)

    def __deepcopy__(self, meme):
        return BuzzKVCacheFast(self.sink_size,
                               self.window_size,
                               self.stride_size,
                               self.k_seq_dim,
                               self.v_seq_dim,
                               self.sample_threshold)

    def clear(self):
        self.cur_sample_size = 0


class BuzzKVCacheWithAccumulation:
    def __init__(
            self,
            sink_size=4,
            window_size=4,
            stride_size=3,
            k_seq_dim=2,
            v_seq_dim=2,
            sample_threshold=180,
    ):
        print(
            f"BuzzKVCache with local max accumulate attn, sink size, stride size, sample threshold, window size: "
            f"{sink_size}, "
            f"{stride_size}, "
            f"{sample_threshold}, "
            f"{window_size}")
        self.sink_size = sink_size
        self.window_size = window_size
        self.stride_size = stride_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.k_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[k_seq_dim]
        self.v_slice_with_max = DIM_TO_SLICE_WITH_LOCAL_MAX[v_seq_dim]
        self.sample_threshold = sample_threshold
        self.cur_sample_size = 0
        self.hh_score = None

    def get_sink_other_window(self, new_coming_words):
        new_coming_size = new_coming_words[0].size(self.k_seq_dim)
        if new_coming_size <= self.sink_size:
            return (new_coming_words,
                    [],
                    [])
        elif new_coming_size <= self.sink_size + self.window_size:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [

                ], [

                    self.k_slice(k, self.sink_size),
                    self.v_slice(v, self.sink_size),
                ])
        else:
            k = new_coming_words[0]
            v = new_coming_words[1]
            return (
                [
                    self.k_slice(k, 0, self.sink_size),
                    self.v_slice(v, 0, self.sink_size),
                ], [
                    self.k_slice(k, self.sink_size, -self.window_size),
                    self.v_slice(v, self.sink_size, -self.window_size),
                ], [
                    self.k_slice(k, -self.window_size),
                    self.v_slice(v, -self.window_size),
                ])

    def concat_three_parts(self, one, two, three):

        concat = []

        part_one_0 = one[0] if len(one) > 0 else None
        part_two_0 = two[0] if len(two) > 0 else None
        part_three_0 = three[0] if len(three) > 0 else None

        parts_0 = [p for p in [part_one_0, part_two_0, part_three_0] if p is not None]
        concat_0 = torch.cat(parts_0, dim=self.k_seq_dim) if len(parts_0) > 0 else None

        part_one_1 = one[1] if len(one) > 1 else None
        part_two_1 = two[1] if len(two) > 1 else None
        part_three_1 = three[1] if len(three) > 1 else None

        parts_1 = [p for p in [part_one_1, part_two_1, part_three_1] if p is not None]
        concat_1 = torch.cat(parts_1, dim=self.k_seq_dim) if len(parts_1) > 0 else None

        if concat_0 is not None:
            concat.append(concat_0)
        if concat_1 is not None:
            concat.append(concat_1)
        return concat

    def sample_extract(self, evicted, stride):
        k = evicted[0]
        v = evicted[1]
        return [
            self.k_slice(k, start=0, step=stride),
            self.v_slice(v, start=0, step=stride),
        ]

    @staticmethod
    def _de_repeat_q(hidden_states, n_de_rep):
        batch, num_of_q_heads, s_len, head_dim = hidden_states.shape
        if num_of_q_heads % n_de_rep != 0:
            raise ValueError(f"Cannot chunk because {num_of_q_heads} % {n_de_rep}")
        chunks = torch.chunk(hidden_states, chunks=n_de_rep, dim=1)
        return torch.sum(torch.stack(chunks, dim=0), dim=0)

    def _update_hh_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]
        if self.hh_score is None:
            """叠在一起表示每个token对其它的token的影响力之和"""
            """hh_score是一个头 * token len的矩阵，代表了在每个头下，每个token收获的注意力"""
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    @staticmethod
    def get_local_max_bool_mask(evicted_score, cur_stride, prob=1):
        head_cnt, _ = evicted_score.shape
        cur_mask = torch.zeros_like(evicted_score, dtype=torch.bool)
        start_idx = 0
        while start_idx < evicted_score.shape[1]:
            end_idx = start_idx + cur_stride
            if end_idx > evicted_score.shape[1]:
                end_idx = evicted_score.shape[1]
            window = evicted_score[:, start_idx:end_idx]
            if random.random() < prob:
                local_max_indices = window.max(dim=1).indices
            else:
                local_max_indices = torch.randint(low=0, high=end_idx - start_idx, size=(head_cnt, 1))
            global_indices = start_idx + local_max_indices
            for i in range(evicted_score.shape[0]):
                cur_mask[i, global_indices[i]] = True
            start_idx = end_idx
        return cur_mask

    def __call__(self, past_key_values, attn_scores):
        # attn_scores = self._de_repeat_q(attn_scores, 8)
        self._update_hh_score(attn_scores)
        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.window_size + self.sink_size:
            return past_key_values
        head_num = past_key_values[0].size(self.k_seq_dim - 1)
        feature_num = past_key_values[0].size(-1)
        sink, other, window = self.get_sink_other_window(past_key_values)
        other_size = other[0].size(self.k_seq_dim)
        if other_size <= self.sample_threshold:
            return past_key_values
        old_sample = [
            self.k_slice(other[0], 0, self.cur_sample_size),
            self.v_slice(other[1], 0, self.cur_sample_size)]
        smaller_stride = (self.stride_size + 1) // 2
        old_sample = self.sample_extract(old_sample, smaller_stride)
        evicted = [
            self.k_slice(other[0], self.cur_sample_size),
            self.v_slice(other[1], self.cur_sample_size)
        ]

        # 在取new sample的时候，要同步更新attn scores
        fake_sink_and_old_sample_attn_score = torch.zeros(head_num,
                                                          self.sink_size +
                                                          (self.cur_sample_size + smaller_stride - 1) // smaller_stride
                                                          , device=past_key_values[0].device)
        evicted_attn_score = self.hh_score[:, self.sink_size + self.cur_sample_size:-self.window_size]
        new_sample_attn_mask = self.get_local_max_bool_mask(evicted_attn_score, self.stride_size)
        new_sample_attn_score = evicted_attn_score[new_sample_attn_mask]
        new_sample_attn_score = new_sample_attn_score.view(head_num, -1)
        window_attn_score = self.hh_score[:, -self.window_size:]
        self.hh_score = torch.cat((fake_sink_and_old_sample_attn_score, new_sample_attn_score, window_attn_score), dim=1)

        new_sample = [evicted[0][:, new_sample_attn_mask, :].view(1, head_num, -1, feature_num),
                      evicted[1][:, new_sample_attn_mask, :].view(1, head_num, -1, feature_num)]
        cur_sample = self.concat_three_parts(old_sample, new_sample, [])
        self.cur_sample_size = cur_sample[0].size(self.k_seq_dim)
        return self.concat_three_parts(sink, cur_sample, window)

    def __deepcopy__(self, meme):
        return BuzzKVCacheWithAccumulation(
            sink_size=self.sink_size,
            window_size=self.window_size,
            stride_size=self.stride_size,
            k_seq_dim=self.k_seq_dim,
            v_seq_dim=self.v_seq_dim,
            sample_threshold=self.sample_threshold
        )

    def clear(self):
        self.hh_score = None
        self.cur_sample_size = 0


class H2OKVCache:
    def __init__(
            self,
            hh_size=120,
            recent_size=120,
            k_seq_dim=2,
            v_seq_dim=2,
    ):
        print(f"H2OKVCache, hh_size, recent size: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache):

        # attn_score_cache = self._de_repeat_q(attn_score_cache, 8)
        """
        1. 第一次问题进来，用attn score evict
        2. 已经在generate了，在past key value上进行叠加
        """
        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        """这是的attn score，去除掉window的，在这里1087 - 180 = 907"""
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        """在这里是得到每个token最大的k个分数的idx"""
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        """这个的作用就是取window，如果有32个token，那就取32个window，然后把window和top k cat在一起"""
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(
            keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        """mask这儿的最终作用，实际上是一个bool矩阵，能够在输入的past key values上取值，在这儿是32"""
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)

        return k_hh_recent, v_hh_recent

    def _update_hh_score(self, attn_score_cache):
        num_new_tokens = attn_score_cache.shape[2]
        if self.hh_score is None:
            """叠在一起表示每个token对其它的token的影响力之和"""
            """hh_score是一个头 * token len的矩阵，代表了在每个头下，每个token收获的注意力"""
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    @staticmethod
    def _de_repeat_q(hidden_states, n_de_rep):
        batch, num_of_q_heads, s_len, head_dim = hidden_states.shape
        if num_of_q_heads % n_de_rep != 0:
            raise ValueError(f"Cannot chunk because {num_of_q_heads} % {n_de_rep}")
        chunks = torch.chunk(hidden_states, chunks=n_de_rep, dim=1)
        return torch.sum(torch.stack(chunks, dim=0), dim=0)

    def __deepcopy__(self, meme):
        return H2OKVCache(hh_size=self.hh_size,
                          recent_size=self.recent_size,
                          k_seq_dim=self.k_seq_dim,
                          v_seq_dim=self.v_seq_dim)

    def clear(self):
        self.hh_score = None

# if __name__ == '__main__':
#     test_cache = BuzzKVCacheWithAccumulation()
#     tensor = torch.randn(1, 32, 5, 5).to("cuda")
#     stride = 3
#     mask = torch.zeros_like(tensor, dtype=torch.bool)
#     test_cache.get_local_max_bool_mask(tensor, stride, prob=0)
