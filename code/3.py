import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim

show = ToPILImage()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])
trainset = tv.datasets.CIFAR10(
    root='./pytorch-book-cifar10/',
    train=True,
    download=True,
    transform=transform
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
testset = tv.datasets.CIFAR10(
    './pytorch-book-cifar10/',
    train=False,
    download=True,
    transform=transform
)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

(data,label) = trainset[100]
print(classes[label])
show((data+1)/2).resize((100,100)).show()

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100)).show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%2000 == 1999:
            print('[%d,%5d] loss: %.3f' \
                  % (epoch+1,i+1,running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
net.to(device)
images = images.to(device)
labels = labels.to(device)
output = net(images)
loss = criterion(output, labels)
print(loss)

dataiter = iter(testloader)
images, labels = next(dataiter)
print(' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100)).show()

outputs = net(images)
_, predicted = t.max(outputs.data, 1)
print(' '.join('%5s'% classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('%f %%'%(100 * correct // total))