import numpy as np
import random

rank_candidates = [2, 4, 6, 8, 10, 12, 16]
rank_list = []

n_layers = 32

# all the same
# for rank in rank_candidates:
#     rank_list.append([rank for i in range(n_layers)])


# random sample
def random_sample(rank_candidates, n_layers):
    ranks = []
    for i in range(n_layers):
        ranks.append(random.choice(rank_candidates))

    return ranks


for i in range(100):
    ranks = random_sample(rank_candidates, n_layers)
    rank_list.append(ranks)
print(np.array(rank_list))
