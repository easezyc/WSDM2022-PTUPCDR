import os
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='3')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.3, 0.7])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ['CDs_and_Vinyl', 'Electronics', 'Grocery_and_Gourmet_Food', 'Video_Games']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]]:
            for task in ['1', '2', '3', '4']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main()
