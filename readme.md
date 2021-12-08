# Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR)
This is the official implementation of our paper **Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR)**, which has been accepted by WSDM2022. [Paper](https://arxiv.org/pdf/2110.11154.pdf)

Cold-start problem is still a very challenging problem in recommender systems. Fortunately, the interactions of the cold-start users in the auxiliary source domain can help cold-start recommendations in the target domain. How to transfer user's preferences from the source domain to the target domain, is the key issue in Cross-domain Recommendation (CDR) which is a promising solution to deal with the cold-start problem. Most existing methods model a common preference bridge to transfer preferences for all users. Intuitively, since preferences vary from user to user, the preference bridges of different users should be different. Along this line, we propose a novel framework named Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR). Specifically, a meta network fed with users' characteristic embeddings is learned to generate personalized bridge functions to achieve personalized transfer of preferences for each user. To learn the meta network stably, we employ a task-oriented optimization procedure. With the meta-generated personalized bridge function, the user's preference embedding in the source domain can be transformed into the target domain, and the transformed user preference embedding can be utilized as the initial embedding for the cold-start user in the target domain.  Using large real-world datasets, we conduct extensive experiments to evaluate the effectiveness of PTUPCDR on both cold-start and warm-start stages.

## Requirements

- Python 3.6
- Pytorch > 1.0
- tensorflow
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json         # Configurations
│   ├── entry.py            # Entry function
│   ├── models.py           # Models based on MF, GMF or Youtube DNN
│   ├── preprocessing.py    # Parsing and Segmentation
│   ├── readme.md
│   └── run.py              # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html) or [Google Drive](https://drive.google.com/drive/folders/1BR4W4eN4nI8aoJw0kvT0_AUC-jD5NeY8?usp=sharing). 
Download the three domains: Music, Movies, Books (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models within `MF, GMF or DNN`, default for `MF`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default for `[0.8, 0.2]`
- epoch: pre-training and CDR mapping training epoches, default for `10`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.01`
- model_name: base model for embedding, default for `MF`

You can run this model through:

```powershell
# Run directly with default parameters 
python entry.py

# Reset training epoch to `10`
python entry.py --epoch 20

# Reset several parameters
python entry.py --gpu 1 --lr 0.02

# Reset seed (we use seed in[900, 1000, 10, 2020, 500])
python entry.py --seed 900
```

If you wanna try different `weight decay`, `meta net dimension`, `embedding dimmension` or more tasks, you may change 
the settings in `./code/config.json`. Note that this repository consists of our PTUPCDR and three baselines, TGTOnly, CMF, and EMCDR.


## Reference

```
Zhu Y, Tang Z, Liu Y, et al. Personalized Transfer of User Preferences for Cross-domain Recommendation[C]. The 15th ACM International Conference on Web Search and Data Mining, 2022.
```

or in bibtex style:

```
@inproceedings{zhu2022ptupcdr,
  title={Personalized Transfer of User Preferences for Cross-domain Recommendation},
  author={Zhu, Yongchun and Tang, Zhenwei and Liu, Yudan and Zhuang, Fuzhen, and Xie, Ruobing and Zhang, Xu and Lin, Leyu and He, Qing},
  inproceedings={The 15th ACM International Conference on Web Search and Data Mining},
  year={2022}
}
```