import copy


class CacheCoordinator:
    def __init__(self, num_of_layer, kv_cache):
        self.num_of_layer = num_of_layer
        self.cur_layer_num = 1
        self.kv_coordinator = {1: kv_cache}
        for i in range(2, num_of_layer + 1):
            self.kv_coordinator[i] = copy.deepcopy(kv_cache)

    def evict(self, past_key_value, attn_scores):
        res = self.kv_coordinator[self.cur_layer_num](past_key_value, attn_scores)
        self.cur_layer_num += 1
        return res

    def cleanup_layers(self):
        self.cur_layer_num = 1

    def cleanup_cache(self):
        for kv_cache in self.kv_coordinator.values():
            kv_cache.clear()
