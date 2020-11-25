# In order to reproduce simulations, I would like to use np.random as a single random number generator
# instead of switching between np.random and random.
# However, random.choices outperforms np.random.choice - especially when using cumulated weights.
# Time to do some time-measurings...

#%%
import random
import numpy as np
#%%
n= 200
x=list(range(n))
p = np.exp(np.random.normal(0,10,len(x)))
p = p/sum(p)
P = np.cumsum(p)

print('n=%d' % n)
print('random.choices(weights)')
%timeit random.choices(x, weights=p) # 4.8 µs (n=10) / 20 µs (n=100)
print('random.choices(cum_weights)')
%timeit random.choices(x, cum_weights=P) # 2 µs (n=10) / 2.5 µs (n=100) / 2.75 µs (n=300)

print('np.random.choice(size=1)')
%timeit np.random.choice(x, size=1, p=p) # 35 µs (n=10) / 40 µs (n=100)
print('np.random.choice(size=10)')
%timeit np.random.choice(x, size=10, p=p) # 36 µs (n=10) / 40 µs (n=100)

%timeit np.random.sample(1) # 1.25 µs

# Observation: 
# n=10:
# - random.choices is much faster than np.random.choice
#   and cumulated weights gains another factor 2
# n=300:
# - np.random.choice is on par with random.choices(weights)
# - random.choices(cum_weights) is still much faster that numpy
# Conclusion:
# - np.random.choice is too slow for my purposes
# - Use a combination of np.random.sample and binary search, see below

#%%
import bisect
def my_#(x, cum_weights):
    p = np.random.sample(1)[0]
    return x[bisect.bisect_left(cum_weights, p, hi=len(cum_weights)-1)]

%timeit my_choice(x,P) # 2.5 µs (n=10) / 2.75 µs (n=100)

import bisect
def my_choice2(x, cum_weights, size):
    p = np.random.sample(size)
    f = lambda y: x[bisect.bisect_left(cum_weights, y, hi=len(cum_weights)-1)]
    return list(map(f,p))

%timeit my_choice2(x,P, size=5) # 7.6 µs (n=10) / 9 µs (n=100)
%timeit my_choice2(x,P, size=10) # 12 µs (n=10) / 15 µs (n=100)

# %%
