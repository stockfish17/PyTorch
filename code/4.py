import torch as t
import numpy as np

a = t.Tensor(2,3)
print(a)
b = t.Tensor([[1,2,3],[4,5,6]])
print(b)
b.tolist()
print(b)
c = t.Tensor(t.rand(2,3))
print(c)
b_size = b.size()
print(b_size)
c = t.Tensor(b_size)
d = t.Tensor((2,3))
print(c, d)

t.Tensor()
a = t.tensor([2,3])
print(a.type())
b = t.Tensor([2,3])
print(b.type())

arr = np.ones((2,3),dtype=np.float64)
a = t.tensor(arr)
print(a)
print(t.ones(2,3))

input_tensor = t.tensor([[1,2,3],[4,5,6]])
t.ones_like(input_tensor)
print(t.zeros(2,3))
print(t.eye(2,3,dtype=t.int))
print(t.arange(1,6,2))
print(t.linspace(1,10,3))
print(t.randn(2,3))
print(t.randperm(5))
a = t.tensor((),dtype=t.int32)
a.new_ones((2,3))
print(a.numel(), a.nelement())

a = t.rand(2,3)
print(a.dtype)
t.set_default_tensor_type('torch.DoubleTensor')
a = t.rand(2,3)
print(a.dtype)
t.set_default_tensor_type('torch.FloatTensor')

b1 = a.type(t.FloatTensor)
b2 = a.float()
b3 = a.type_as(b1)
print(a.dtype,b1.dtype,b2.dtype,b3.dtype)

a.new_ones(2,4)
a = t.randn(2,3).cuda()
a.new_ones(2,4)
print(a)

a = t.randn(3,4)
print(a)
print(a[0])
print(a[:,1])
print(a[1,-2:])
print((a>0))
print((a>0).int())

print(a[a>0])
print(a.masked_select(a>0))
print(t.where(a>0,a,t.zeros_like(a)))

a = t.arange(0,16).view(4,4)
print(a)
index = t.tensor([[0,1,2,3]])
a.gather(0,index)
index = t.tensor([[3,2,1,0]]).t()
a.gather(1,index)

index = t.tensor([[0,1,2,3],[3,2,1,0]]).t()
b = a.gather(1,index)

c = t.zeros(4,4).long()
c.scatter_(1,index,b)

t.Tensor([1.]).item()

a = t.arange(6).view(2,3)
print(a)
print(t.cat((a,a),0))
print(t.cat((a,a),1))
b = t.stack((a,a),0)
print(b)
print(b.shape)

x = t.arange(0,16).view(2,2,4)
print(x)
print(x[[1,0],[1,1],[2,0]])
print(x[[1,0],[0],[1]])

a = t.arange(0,6).float().view(2,3)
t.cos(a)
print(a%3)
print(t.fmod(a,3))
print(a)
print(t.clamp(a,min=2,max=4))
