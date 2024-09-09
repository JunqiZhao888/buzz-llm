import torch
import numpy as np
import random


def slice1d(x, start, end=None, step=1):
    return x[:, start:end:step, ...]


def slice2d(x, start, end=None, step=1):
    return x[:, :, start:end:step, ...]


def slice3d(x, start, end=None, step=1):
    raise ValueError("need further check, 3rd dimension kv not supported")
    # return x[:, :, :, start:end:step, ...]


def slice1d_with_local_max(x, start, end=None, step=1, attn_scores=None, probability=0.5):
    raise ValueError("need further check, 3rd dimension kv not supported")


def slice3d_with_local_max(x, start, end=None, step=1, attn_scores=None, probability=0.5):
    raise ValueError("need further check, 3rd dimension kv not supported")


def slice2d_with_local_max(x, start, end=None, step=1, attn_scores=None, probability=0.5):
    if attn_scores is None or len(attn_scores) == 0:
        raise ValueError("Attn scores cannot be empty")
    if end is None:
        end = len(x[0][0])
        if end == 0:
            return x
    result = []
    batch_num, head_num, _, feature_num = x.shape
    if batch_num > 1:
        raise ValueError(f"Unexpected batch size {batch_num}")
    for i in range(start, end, step):
        if np.random.rand() < probability:
            current_end = min(i + step, end)
            # TODO 问题就在这个地方，假设没有window，那attention score应该是去掉了sink和old sample之后再开始计算
            cur_scores = attn_scores[:, i:current_end]
            _, max_indices = torch.max(cur_scores, dim=1)
            selected_tokens = x[0, torch.arange(head_num), i + max_indices, :]
            selected_tokens = selected_tokens.view(batch_num, head_num, 1, feature_num)
            result.append(selected_tokens)
        else:
            result.append(x[:, :, i:i + 1, ...])  # 取当前采样位置的值，保持维度一致
    return torch.cat(result, dim=2)


def get_local_max_bool_mask(self, attn_score, cur_stride, prob=1):
    attn_score = self.de_repeat_q(attn_score, 8)
    attn_score = attn_score.sum(0).sum(1)
    head_cnt, _ = attn_score.shape
    evicted_score = attn_score[:, self.sink_size + self.cur_sample_size:-self.window_size]
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


# if __name__ == '__main__':
#     print()
    # """测试概率找到stride内的最大值"""
    # torch.set_printoptions(sci_mode=False)
    # # kv: batch head token feature
    # tokens = torch.randn(1, 2, 5, 3)
    # # batch head query kv -> batch kv
    # score: Tensor = torch.randn(2, 10)
    # score[0, 2] = 30
    # score[1, 3] = 30
    # slice2d_with_local_max(x=tokens, start=0, step=3, probability=1, attn_scores=score)
    #
    # """测试概率找到stride内的最大值，但是是使用mask的办法"""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = 5
    # tensor = torch.randn(4, x).to(device)
    # stride = 3
    # mask = torch.zeros_like(tensor, dtype=torch.bool)
    # start_idx = 0
    # while start_idx < tensor.shape[1]:
    #     end_idx = start_idx + stride
    #     if end_idx > tensor.shape[1]:
    #         end_idx = tensor.shape[1]
    #     # 取出当前窗口的数据
    #     window = tensor[:, start_idx:end_idx]
    #     # 计算每个窗口的最大值索引
    #     local_max_indices = window.max(dim=1).indices
    #     # 计算全局索引
    #     global_indices = start_idx + local_max_indices
    #     # 更新布尔掩码
    #     for i in range(tensor.shape[0]):
    #         mask[i, global_indices[i]] = True
    #     start_idx = end_idx
