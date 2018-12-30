# -*- coding: utf-8 -*-
import numpy as np
x = [1, 2, 3]
y = [5, 4, 6]
d = zip(x, y)

c = enumerate(d)

f = {k:max(a1, b1) for k,(a1, b1) in c}

s = sorted(f.keys(), key = lambda x:f[x])
