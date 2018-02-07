# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from Model import PoemGenerator
from config import Config
from data import PoemDataset
import argparse

torch.manual_seed(123)



parser = argparse.ArgumentParser()

parser.add_argument('--word_embedding', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--dict_size', type=int, default=8293)
parser.add_argument('--training_path', type=str, default='./data/tang.npz')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--check_path', type=str, default='./checkpoints/')
parser.add_argument('--print_step', type=int, default=2, help='How many step then print the loss')

args = parser.parse_args()

# 配置参数
config = Config(word_embedding=args.word_embedding,
                dict_size=args.dict_size,
                batch_first=True,
                bidirectional=True,
                training_set=args.training_path,
                batch_size=args.batch_size,
                cuda=args.gpu,
                epoch=args.epoch)

model = PoemGenerator(dict_size=config.dict_size,
                      word_embedding=config.word_embedding,
                      batch_first=True)

if torch.cuda.is_available():
    torch.cuda.set_device(config.cuda)
    Data = PoemDataset(config.training_set, cuda=True)
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size,
                               pin_memory=True, shuffle=True)
    model = model.cuda()
else:
    Data = PoemDataset(config.training_set)
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

for epoch in range(config.epoch):
    for step, x in enumerate(DataIter):

        # Training data & Labels
        train_set = autograd.Variable(x[:, 1:])
        labels = autograd.Variable(x[:, :-1]).contiguous().view(-1)

        out = model(train_set)
        out = out.view(-1, config.dict_size)

        loss = criterion(out, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_step == 0:
            print("[epoch %d, step %d] Loss: %.11f" % (epoch, step, loss))
            print(model.generating_acrostic_poetry('黄总牛逼', Data))
            print(model.generating_acrostic_poetry('龙眼爆世强', Data))

    torch.save(model.state_dict(), args.check_path+str(epoch)+'.ckpt')







