import numpy as np
import time
from gibbs import gibbs_jointpredupdt

P0 = np.array([[0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.345108847352739, np.inf, np.inf],
               [np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.849090957754662, np.inf],
               [np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, 1.64038243547480],
               [np.inf, np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf]])
#P1 = np.array([[0.03045921, 7.4185809, -1.1376525]])

test_num = 1000
start = time.time()
for i in range(test_num):
    assignments, costs = gibbs_jointpredupdt(P0, 1000)
print((time.time() - start) / test_num)
for assignment, cost in zip(assignments, costs):
    print(assignment, cost)
