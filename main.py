# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from model import PoemGenerator
from config import Config
from data import PoemDataset
import argparse
import pickle
import visdom

torch.manual_seed(11)

# 显存泄露的时候使用下面语句（PyTorch中LSTM的祖传BUG）
#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()

parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--word_embedding', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--dict_size', type=int, default=8293)
parser.add_argument('--training_path', type=str, default='./data/tang.npz')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--check_path', type=str, default='./checkpoints/')
parser.add_argument('--print_step', type=int, default=2, help='How many step then print the loss')
parser.add_argument('--rnn', type=str, default='LSTM')
parser.add_argument('--num_layers', type=int, default=2)

args = parser.parse_args()

# 配置参数
config = Config(word_embedding=args.word_embedding,
                dict_size=args.dict_size,
                batch_first=True,
                bidirectional=args.bidirectional,
                training_path=args.training_path,
                batch_size=args.batch_size,
                cuda=args.gpu,
                epoch=args.epoch,
                rnn=args.rnn)

model = PoemGenerator(dict_size=config.dict_size,
                      word_embedding=config.word_embedding,
                      batch_first=True,
                      rnn=config.rnn,
                      bidirectional=config.bidirectional,
                      num_layers=config.num_layers)

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    torch.cuda.set_device(config.cuda)
    Data = PoemDataset(config.training_path)
    model.cuda()
    criterion.cuda()
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size, shuffle=True, pin_memory=True)

else:
    Data = PoemDataset(config.training_path)
    DataIter = data.DataLoader(dataset=Data, batch_size=config.batch_size, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=config.lr)

loss_list = []

vis = visdom.Visdom(env='poem')
vis.text('The loss list:', win='loss', env='poem')
vis.text(model.generating_acrostic_poetry('李星熠蒋桂达', Data), env='poem', win='loss2')
ii = 0
print(ii)
vis.line(X=torch.FloatTensor([ii]), Y=torch.FloatTensor([ii]), env='poem', win='LOSS', update='append' if ii > 0 else None)

for epoch in range(config.epoch):
    for step, x in enumerate(DataIter):

        x = x.contiguous()
        # Training data & Labels
        if torch.cuda.is_available():
            train_set = x[:, :-1].cuda()
            labels = x[:, 1:].contiguous().view(-1).cuda()
        else:
            train_set = x[:, :-1]
            labels = x[:, 1:].contiguous().view(-1)

        train_set = autograd.Variable(train_set)
        labels = autograd.Variable(labels)

        out = model(train_set)
        out = out.view(-1, config.dict_size)

        loss = criterion(out, labels)
#        loss_list.append(loss)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_step == 0:
            out = model.generating_acrostic_poetry('李星熠蒋桂达', Data)
            print(out)
            vis.text("[epoch %d, step %d] Loss: %.11f" % (epoch, step, loss), win='loss', env='poem', append=True)
            vis.text(out, win='loss2', append=True)
            vis.line(X=torch.FloatTensor([ii]), Y=loss.data, env='poem', win='LOSS', update='append' if ii != 0 else None)
            ii += 1

#    torch.save(model.state_dict(), f=args.check_path+str(epoch)+'.ckpt')

    with open('data/loss.pkl', 'wb') as f:
        pickle.dump(loss_list, f)

