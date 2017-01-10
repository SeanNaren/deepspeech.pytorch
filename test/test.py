import torch
from torch.autograd import Variable

from CTCLoss import ctc_loss

loss = ctc_loss()
input = Variable(torch.FloatTensor([[[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]]).transpose(0, 1))
target = Variable(torch.FloatTensor([[3, 3]]))
sizes = Variable(torch.FloatTensor(input.size(1)).fill_(input.size(0)))
print loss(input, target, sizes)

input = Variable(torch.FloatTensor([[[-5,-4,-3,-2,-1], [-10,-9,-8,-7,-6], [-15,-14,-13,-12,-11]]]).transpose(0, 1))
target = Variable(torch.FloatTensor([[2, 3]]))
sizes = Variable(torch.FloatTensor(input.size(1)).fill_(input.size(0)))
print loss(input, target, sizes)

input = Variable(torch.FloatTensor([
      [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
      [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],
      [[-5,-4,-3,-2,-1],[-10,-9,-8,-7,-6],[-15,-14,-13,-12,-11]]
   ]).transpose(0, 1))
target = Variable(torch.FloatTensor([[1,0],[3,3],[2,3]]))
sizes = Variable(torch.FloatTensor([1, 3, 3]))
print torch.sum(loss(input, target, sizes))