from __future__ import print_function
import torch as T

a = T.empty(5, 3)
# print (a)

b = T.rand(5, 3)
# print (b)

c = T.zeros(5, 3, dtype=T.int)
# print (c)

d = T.tensor([5.5, 3])
# print (d)

e = d.new_ones(5, 3, dtype=T.double)
# print (e)

f = T.randn_like(d, dtype=T.float)
# print (f)

# print (b.size())

x = T.rand(5, 3)
y = T.rand(5, 3)
# print (x + y)

# print (T.add(x, y))

result = T.empty(5, 3)
T.add(x, y, out=result)
# print (result)

y.add_(x)
# print (x)

# print (x[:, 1])

x = T.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# print (x.size(), y.size(), z.size())

x = T.randn(1)
# print (x)
# print (x.item())

a = T.ones(5)
print (a)
