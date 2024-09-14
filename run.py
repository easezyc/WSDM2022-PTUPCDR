import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import tqdm
from tensorflow import keras
from models import MFBasedModel, GMFBasedModel, DNNBasedModel

class Run():
    def __init__(self,
                 config
                 ):
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']
        
        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'aug_mae': 10, 'aug_rmse': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10}
        
        self.filename = config['filename']

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        print("Reading CSV file from path:", path)
        if not history:
            print("Not History Part")
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            print("History Part")
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            #data_iter = DataLoader(dataset, 21, shuffle=False)

            return data_iter

    def read_map_data(self):
        #cols = ['uid', 'iid', 'y', 'd1', 'd2', 'pos_seq']
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        #cols_train = ['uid', 'iid', 'y', 'd1', 'd2']
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'DNN':
            model = DNNBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'GMF':
            model = GMFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        print("Predicts: ", str(predicts))
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
   

    def calculate_metrics_and_save_topk(self, batch_results, topks=[5, 10], filename_prefix="metrics_per_uid"):
        for topk in topks:
            metrics_list = []
    
            for uid, data in batch_results.items():
                y = np.array(data['y'])
                pred = np.array(data['pred'])

                # Only one valid index (where y != -1)
                valid_index = np.where(y != -1)[0][0]
                sorted_indices = np.argsort(pred)[::-1]  # Indices of predictions sorted descending
                rank = np.where(sorted_indices == valid_index)[0][0] + 1  # 1-based index

                # Calculate metrics
                mrr = 1.0 / rank if rank <= topk else 0
                hit = 1 if rank <= topk else 0
                ndcg = (1 / np.log2(rank + 1)) if rank <= topk else 0
    
                metrics_list.append([uid, mrr, hit, ndcg])
    
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_list, columns=["UID", "MRR", "HIT", "NDCG"])
            
            # Calculate and print average metrics
            avg_mrr = metrics_df['MRR'].mean()
            avg_hit = metrics_df['HIT'].mean()
            avg_ndcg = metrics_df['NDCG'].mean()
            print(f"Top-{topk} Average HIT: {avg_hit:.4f}, Average NDCG: {avg_ndcg:.4f}, Average MRR: {avg_mrr:.4f}")

            # Save to CSV
            filename = f"{filename_prefix}_top{topk}.csv"
            metrics_df.to_csv(filename, index=False)

            # Optionally, return the average metrics for the last topk calculated
        return avg_mrr, avg_hit, avg_ndcg


    def eval_mae_rating_ranking(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        batch_results = {}  # Initialize batch_results dictionary
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)

                # Extract uids from the first column of X for the current batch
                uids = X[:, 0]
    
                # Loop through the current batch to store results
                for i in range(X.shape[0]):
                    uid_item = uids[i].item()  # Extract uid for the current record
                    pred_item = pred[i].tolist()  # Convert prediction to list (or scalar)
                    y_item = y[i].item()  # Convert y to scalar
                    x_item = X[i].tolist()  # Convert X to list

                    # Store in the dictionary
                    if uid_item not in batch_results:
                        batch_results[uid_item] = {'X': [], 'y': [], 'pred': []}
                    batch_results[uid_item]['X'].append(x_item)
                    batch_results[uid_item]['y'].append(y_item)
                    batch_results[uid_item]['pred'].append(pred_item)

                targets.extend(y.tolist())  # Adjusted for compatibility
                predicts.extend(pred.tolist())

        # Convert targets and predicts lists to tensors for calculation
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts).float()  # Ensure this matches the dtype of predictions

        #self.calculate_metrics_and_save(batch_results, filename="metrics_per_uid.csv")
        self.calculate_metrics_and_save_topk(batch_results, topks=[5, 10], filename_prefix="model_metrics")
        
        """
        # Print the batch_results
        print("\nBatch Results:")
        counter = 0  # Initialize a counter to track the number of items printed

        for uid, results in batch_results.items():
            if counter < 5:  # Check if less than 5 items have been printed
                print(f"UID: {uid}, X: {results['X']}, y: {results['y']}, Pred: {results['pred']}")
                counter += 1  # Increment the counter
            else:
                break  # Break the loop after printing 5 items
        
        #for uid, results in batch_results.items():
         #   print(f"UID: {uid}, X: {results['X']}, y: {results['y']}, Pred: {results['pred']}")
        """
        # Calculate and return the loss (MAE) and RMSE
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
    
    
    def eval_mae_save(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        # Initialize lists to store UIDs, IIDs, targets, and predictions
        uids, iids, targets, predicts = [], [], [], []
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
    
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                # Extract UIDs and IIDs for each row in the batch
                batch_uids = X[:, 0].tolist()
                batch_iids = X[:, 1].tolist()
                uids.extend(batch_uids)
                iids.extend(batch_iids)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())

        # Convert lists to PyTorch tensors for loss computation
        targets_tensor = torch.tensor(targets).float()
        predicts_tensor = torch.tensor(predicts)
    
        # After collecting all data, create a DataFrame and save it to CSV
        df = pd.DataFrame({
            'UID': uids,
            'IID': iids,
            'Targets': targets,
            'Predicts': predicts
        })

        # Sort the DataFrame by UID to group data by user
        df_sorted = df.sort_values(by='UID')

        # Specify your desired path for the CSV file
        csv_file_path = '/scratch/dmeher/Research-PTUPCDR/predictions_targets_uid_iid.csv'
        df_sorted.to_csv(csv_file_path, index=False)
        print(f"Saved grouped predictions and targets by UID to {csv_file_path}")

        print("Predicts: ", str(predicts_tensor))
        # Return the MAE and RMSE
        return loss(targets_tensor, predicts_tensor).item(), torch.sqrt(mse_loss(targets_tensor, predicts_tensor)).item()
 

    def eval_mae_1000(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        total_data_points = 0  # Counter for the number of data points processed
    
        with torch.no_grad():
            counter = 0
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                print("X:", X, "y:", y)
                pred = model(X, stage)
                batch_size = y.size(0)  # Get the batch size
            
                # Adjust the batch size if it exceeds the remaining number of data points to process
                if total_data_points + batch_size > 50:
                    excess = (total_data_points + batch_size) - 50
                    y = y[:-excess]
                    pred = pred[:-excess]
            
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
            
                total_data_points += batch_size
                if total_data_points >= 50:
                    break  # Stop after processing 1000 data points
                
                #if counter < 4:  # Check if the counter is less than 4
                  #  print(f"X[{counter}]:\n{X}")
                 #   print(f"y[{counter}]:\n{y}\n")

                #counter += 1  # Increment counter after each iteration

        # Print the length of targets and predicts to confirm the number of data points
        print(f"Niumber of data points evaluated: {len(predicts)}")
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        print("Targets: ", str(targets))
        print("Predicts: ", str(predicts))
        # Optionally, confirm the exact number here as well
        print(f"Length of targets: {len(targets)}, Length of predicts: {len(predicts)}")

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
    def eval_mae_last_epochs(self, model, data_loader, algo, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        
        user_targets_dict = {}
        user_predicts_dict = {}
        user_rmse_dict = {}  # Dictionary to store UID and RMSE values
        user_ids = list()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
                user_ids.extend(X[:,0].tolist())
            targets = torch.tensor(targets).float()
            predicts = torch.tensor(predicts)

            print("Length of Data_loaders: ", len(data_loader))
            print("Length targets: ", len(targets))
            print("length of user_ids: ", len(user_ids))
            print("Length of unique user_ids", len(set(user_ids)))
        
            # Calculate RMSE for each user
            for i,user_id in enumerate(user_ids):
                user_targets = targets[i]
                user_predicts = predicts[i]
                if user_id not in user_targets_dict:
                    user_targets_dict[user_id] = []
                    user_predicts_dict[user_id] = []
                user_targets_dict[user_id].append(user_targets) 
                user_predicts_dict[user_id].append(user_predicts)

            for user_id, user_target in user_targets_dict.items():
                user_predict = user_predicts_dict[user_id]
                user_rmse_ind = np.sqrt(mse_loss(torch.stack(user_target), torch.stack(user_predict))).item()
                user_mae_ind = loss(torch.stack(user_target), torch.stack(user_predict)).item()
                user_rmse_dict[user_id] = [user_mae_ind, user_rmse_ind]

            
            # Remove '_ptudata' from both src and tgt
            src_cleaned = self.src.replace('_ptudata', '')
            tgt_cleaned = self.tgt.replace('_ptudata', '')
    
            # Store UID and RMSE values in a CSV file
            path = "./results_cs798/"
            filename = path + src_cleaned + tgt_cleaned + "_" + algo + "_" + self.filename
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['UID', 'MAE', 'RMSE'])  # Write header
                for uid, rmse in user_rmse_dict.items():
                    writer.writerow([uid, rmse[0],rmse[1]])  # Write UID and RMSE values
        
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
    
    def getDataRevised(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X, stage)
                loss = criterion(pred, y.squeeze().float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            #===========ORIGINAL CODE======================
            #mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            #==============================================
            #===========RANKING MATRIX FOR LLMasRec PROJECT==============
            #mae, rmse = self.eval_mae_rating_ranking(model, data_test, stage='test_tgt')
            #============================================================
            if i==self.epoch-1:
                mae, rmse = self.eval_mae_last_epochs(model, data_test,'tgt', stage='test_tgt')
            else:
                mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            
            self.update_results(mae, rmse, 'tgt')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print('=========DataAug========')
        for i in range(self.epoch):
            self.train(data_aug, model, criterion, optimizer, i, stage='train_aug')
            #=============ORIGINAL CODE==============
            #mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            #========================================
            #=============RANKING MATRIX FOR LLMasREC PROJECT====================
            #mae, rmse = self.eval_mae_rating_ranking(model, data_test, stage='test_aug')
            #====================================================================
            if i== self.epoch-1:
                mae, rmse = self.eval_mae_last_epochs(model, data_test,'cmf', stage='test_aug')
            else:
                mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            self.update_results(mae, rmse, 'aug')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def CDR(self, model, data_src, data_map, data_meta, data_test,
            criterion, optimizer_src, optimizer_map, optimizer_meta):
        print('=====CDR Pretraining=====')
        for i in range(self.epoch):
            self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')
        print('==========EMCDR==========')
        for i in range(self.epoch): 
            self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
            #===============ORIGINAL CODE========================
            #mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            #====================================================
            #===============RANKING MATRIX FOR LLMasRec PROJECT========================
            #mae, rmse = self.eval_mae_rating_ranking(model, data_test, stage='test_map')
            #====================================================

            if i == self.epoch-1:
                mae, rmse = self.eval_mae_last_epochs(model, data_test,'emcdr', stage='test_map')
            else:
                mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'emcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))
        
        print('==========PTUPCDR==========')
        for i in range(self.epoch):#self.epoch
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta')
            
            if i == self.epoch - 1:
                mae, rmse = self.eval_mae_last_epochs(model, data_test,'ptupcdr', stage='test_meta')
            else:
                mae, rmse = self.eval_mae(model, data_test, stage='test_meta') 
            #=============RANKING MATRIX FOR LLMAsRec Project=====================
            #mae, rmse = self.eval_mae_rating_ranking(model, data_test, stage='test_meta')
            #=====================================================================
            self.update_results(mae, rmse, 'ptupcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))


    def main(self):
        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test = self.get_data()

        ##### Extra Code ####
        # # # Read the dataset from the original CSV file
        # data = pd.read_csv('user_rmse_8_2_trial_2.csv')

        # # Sort the dataset by the 0th column
        # sorted_data = data.sort_values(by=data.columns[0])
        # # Get the unique values in the first column
        # unique_values = data.iloc[:, 0].unique()

        # mean1 = data["RMSE"].mean()
        # mean2 = data["MAE"].mean()
        # print("mean1: ", mean1)
        # print("mean2: ", mean2)
        # print("Data Values count: ", len(data))
        # print("Unique Values:", len(unique_values))

        # # # # Store the sorted dataset in a new CSV file
        # # sorted_data.to_csv('sorted_user_rmse_8_2.csv', index=False)
        ##### End Extra Code ####
        
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()
        self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
        self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
        self.CDR(model, data_src, data_map, data_meta, data_test,
                 criterion, optimizer_src, optimizer_map, optimizer_meta)
        
        print(self.results)
