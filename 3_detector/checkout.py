import config as cf
import numpy as np

mean = cf.mean
std = cf.std

mean = map(lambda x : -x, mean)
std = map(lambda x : 1/x, std)

print(mean)
print(std)
