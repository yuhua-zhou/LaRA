import os
print(os.environ.get('HF_HOME'))

# import time
# import datetime
# import torch
#
# count = 0
#
# while True:
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     count += 1
#     a = torch.randn(3, 4, device=device)
#     b = torch.randn(4, 3, device=device)
#     c = a @ b
#     print('%s: count: %d \r' % (datetime.datetime.now(), count))
#     time.sleep(1800)
