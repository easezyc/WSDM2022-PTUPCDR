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
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.8,0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--filename', type=str, default='output.csv', help='Name of the file for output')
    # New argument for test data file path
    parser.add_argument('--test_data', type=str, default='test.csv',required=False, help='Path to the test data file')
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
        config['filename'] = args.filename  # Add the filename to the config
        config['test_data']=args.test_data
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        '''
        # ORIGINAL CODE
        for dealing in ['CDs_and_Vinyl', 'Electronics', 'Grocery_and_Gourmet_Food', 'Books', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()
        '''
        
        # FOR USER CORRELATION PROJECT
        '''
        #FOR BOOK MOVIE MUSIC PAIR
        for dealing in ['book_ptudata', 'movie_ptudata', 'music_ptudata']:
            DataPreprocessingMid(config['root'], dealing).main()
        '''
        for dealing in ['electronic_ptudata', 'food_ptudata', 'videogame_ptudata']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        '''
        # FOR ORIGINAL CODE
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]]:
            for task in ['1', '2', '3']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))
        '''
        # FOR USER CORRELATION PROJECT
        for ratio in [[0.8, 0.2]]:
            for task in ['1', '2']:
                DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio, config['test_data']).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))
        
    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main()
