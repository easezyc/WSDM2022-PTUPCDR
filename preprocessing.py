import pandas as pd
import gzip
import json
import tqdm
import random
import os

class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing
        
    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                #==============FOR AMAZON 5 CORE DATA =======================
                #re.append([line['reviewerID'], line['asin'], line['overall']])
                #============================================================≈

                #==============FOR AMAZON REVIEW 23 =======================
                #re.append([line['user_id'], line['asin'], line['rating']])
                re.append([line['uid'], line['iid'], line['y']])
                #============================================================≈
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

class DataPreprocessingReady():
    def __init__(self,
                 root,
                 src_tgt_pairs,
                 task,
                 ratio, test_data):
        self.root = root
        self.src = src_tgt_pairs[task]['src']
        self.tgt = src_tgt_pairs[task]['tgt']
        self.ratio = ratio
        self.test_data=test_data
        
    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    '''
    #========ORIGINAL CODE===========================
    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt
    
    '''

    def mapper(self, src, tgt, test_data):
        # Print information about the source and target datasets
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
    
        # Find common and all unique user IDs
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
    
        # Create mappings for users and items
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))

        # Map the user and item IDs in src and tgt datasets
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
    
        # Map the test data using the same dictionaries as target data (since it's a subset of tgt)
        test_data.uid = test_data.uid.map(uid_dict)
        test_data.iid = test_data.iid.map(iid_dict_tgt)

        # Return the mapped src, tgt, and test_data
        return src, tgt, test_data
    
    def get_history(self, data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_seq_dict[uid] = pos
        return pos_seq_dict

    '''
    #=====ORIGINAL CODE===========================
    def split(self, src, tgt):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        co_users = src_users & tgt_users
        test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
        test = tgt[tgt['uid'].isin(test_users)]
        pos_seq_dict = self.get_history(src, co_users)
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
        train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
        test['pos_seq'] = test['uid'].map(pos_seq_dict)
        return train_src, train_tgt, train_meta, test
    '''
    
    def split(self, src, tgt, test_data):
        # Print the total number of unique items across both datasets
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
    
        # Get the set of unique users from src and tgt datasets
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
    
        # Find the users common to both datasets
        co_users = src_users & tgt_users
    
        # Use the provided test_data to extract test users
        test_users = set(test_data.uid.unique())  # Extract test users from the provided test_data
    
        # Split the target dataset into train and test sets using the provided test_data
        train_src = src  # The source data remains the same
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]  # Train target data excluding the test users
        test_data = test_data  # Test data as provided
        
        # Generate user history (pos_seq_dict) from src for common users
        pos_seq_dict = self.get_history(src, co_users)
        
        # Map user history (pos_seq) to both train_meta and test_data
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)].copy()  # Meta train data for common users, excluding test users
        train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)  # Map user history to train_meta
        #train_meta.loc[:, 'pos_seq'] = train_meta['uid'].map(pos_seq_dict)

        test_data['pos_seq'] = test_data['uid'].map(pos_seq_dict)  # Map user history to test data
    
        # Return the split datasets: train_src, train_tgt, train_meta, and test_data
        return train_src, train_tgt, train_meta, test_data

    def save(self, train_src, train_tgt, train_meta, test):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root +  '/train_meta.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)

    def main(self):
        '''
        #======ORIGINAL CODE===============
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt = self.mapper(src, tgt)
        train_src, train_tgt, train_meta, test = self.split(src, tgt)
        self.save(train_src, train_tgt, train_meta, test)
        '''
        
        # Step 1: Read the source and target data
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
    
        # Step 2: Read the test data
        test_data = self.read_mid(self.test_data)  # Assuming test_data file path is set elsewhere
    
        # Step 3: Map the source, target, and test data using the mapper function
        src, tgt, test_data = self.mapper(src, tgt, test_data)
    
        # Step 4: Split the data using the provided test data
        train_src, train_tgt, train_meta, test_data = self.split(src, tgt, test_data)
    
        # Step 5: Save the train and test data
        self.save(train_src, train_tgt, train_meta, test_data)
