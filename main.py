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
import pickle

torch.manual_seed(123)

# 显存泄露的时候使用下面语句
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()

parser.add_argument('--word_embedding', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
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
    model = model.cuda()
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size, shuffle=True, pin_memory=True)

else:
    Data = PoemDataset(config.training_set)
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

loss_list = []

for epoch in range(config.epoch):
    for step, x in enumerate(DataIter):

        # Training data & Labels
        if torch.cuda.is_available():
            train_set = x[:, 1:].cuda()
            labels = x[:, :-1].contiguous().view(-1).cuda()
        else:
            train_set = x[:, 1:]
            labels = x[:, :-1].contiguous().view(-1)

        train_set = autograd.Variable(train_set)
        labels = autograd.Variable(labels)

        out = model(train_set)
        out = out.view(-1, config.dict_size)

        loss = criterion(out, labels)
        loss_list.append(loss)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_step == 0:
            print("[epoch %d, step %d] Loss: %.11f" % (epoch, step, loss))

    torch.save(model.state_dict(), f=args.check_path+str(epoch)+'.ckpt')
    print(model.generating_acrostic_poetry('龙眼爆石墙', Data))

    with open('data/loss.pkl', 'wb') as f:
        pickle.dump(loss_list, f)

