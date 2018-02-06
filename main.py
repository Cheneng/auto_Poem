# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from config import Config
from data import PoemDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--word_embedding', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--training_path', type=str, default='data/tang.npz')

args = parser.parse_args()

# 配置参数
config = Config(word_embedding=args.word_embedding,
                dict_size=1000,
                batch_first=True,
                bidirectional=True,
                training_set=args.training_path,
                batch_size=args.batch_size)

Data = PoemDataset(config.training_set)
Data = data.DataLoader(dataset=Data, batch_size=config.batch_size)

