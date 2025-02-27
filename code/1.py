import torch as t
import numpy as np

# x = t.rand(2,3)
# print(x)
# print(x.shape)
# print(x.size()[1])
# print(x.size(1))
# print(x.size(0))

# y = t.rand(2,3)
# print(x+y)
# print(t.add(x,y))
# result = t.Tensor(2,3)
# t.add(x,y,out=result)
# print(result)

# print("-------------------")
# y.add(x)
# print(y)
# y.add_(x)
# print(y)

# a = t.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a = np.ones(5)
# b = t.from_numpy(a)
# print(a)
# print(b)

# b.add_(1)
# print(b)
# print(a)

# scalar = b[0]
# print(scalar)
# print(scalar.shape)
# print(scalar.item())

# tensor = t.tensor([2])
# print(tensor)
# print(tensor.size())
# print(tensor.item())

# tensor = t.tensor([3,4])
# old_tensor = tensor
# new_tensor = old_tensor.clone()
# new_tensor[0] = 1111
# print(old_tensor)
# print(new_tensor)

# new_tensor = old_tensor.detach()
# new_tensor[0] = 1111
# print(old_tensor)
# print(new_tensor)

# x = t.randn(4,4)
# y = x.view(16)
# z = x.view(-1,8)
# print(x.size(), y.size(),z.size())

# p = x.reshape(-1,8)
# print(p.shape)

# x1 = t.randn(2,4,6)
# o1 = x1.permute(2,1,0)
# o2 = x1.transpose(0,2)
# print(f'o1 size {o1.size()}')
# print(f'o2 size {o2.size()}')

# x = t.randn(3,2,1,1)
# y = x.squeeze(-1)
# z = x.unsqueeze(0)
# w = t.cat((x,x),0)
# print(f'y size {y.size()}')
# print(f'z size {z.size()}')
# print(f'w size {w.size()}')

# device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# x = x.to(device)
# y = y.to(x.device)
# z = x + y

# x = t.ones(2,2,requires_grad=True)
# print(x)
# y = x.sum()
# print(y)
# print(y.grad_fn)
# y.backward()
# print(x.grad)

# y.backward()
# print(x.grad)

# y.backward()
# print(x.grad)

a = t.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

