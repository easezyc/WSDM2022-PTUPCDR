# Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR)
This is the official implementation of our paper **Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR)**, which has been accepted by WSDM2022. [Paper](https://dl.acm.org/doi/pdf/10.1145/3488560.3498392?casa_token=rIw_LyxVHwEAAAAA:8NqNOulu_0ML6iQB2f1tgqHrQMT8Okv0Cm7gZmWbCsCzd7b1ZIc-QlHXQ9b8Dj6NTBqnrc415tEi)

Cold-start problem is still a very challenging problem in recommender systems. Fortunately, the interactions of the cold-start users in the auxiliary source domain can help cold-start recommendations in the target domain. How to transfer user's preferences from the source domain to the target domain, is the key issue in Cross-domain Recommendation (CDR) which is a promising solution to deal with the cold-start problem. Most existing methods model a common preference bridge to transfer preferences for all users. Intuitively, since preferences vary from user to user, the preference bridges of different users should be different. Along this line, we propose a novel framework named Personalized Transfer of User Preferences for Cross-domain Recommendation (PTUPCDR). Specifically, a meta network fed with users' characteristic embeddings is learned to generate personalized bridge functions to achieve personalized transfer of preferences for each user. To learn the meta network stably, we employ a task-oriented optimization procedure. With the meta-generated personalized bridge function, the user's preference embedding in the source domain can be transformed into the target domain, and the transformed user preference embedding can be utilized as the initial embedding for the cold-start user in the target domain.  Using large real-world datasets, we conduct extensive experiments to evaluate the effectiveness of PTUPCDR on both cold-start and warm-start stages.

## Introduction
This repository provides the implementations of PTUPCDR and three popular baselines (TGTOnly, CMF, EMCDR):
* TGTOnly：Train a MF model with the data of the target domain.
* CMF: [Relational Learning via Collective Matrix Factorization Categories and Subject Descriptors](https://dl.acm.org/doi/pdf/10.1145/1401890.1401969?casa_token=S9kvmlp1bxEAAAAA:v96uHthvspO1ahgCZ1htH8sGl2voMvREqwXVYGf3X4WbvYXaD7tX1OsfXhx4k126HSOOtsbcbf9q) (KDD 2008)
* EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/Proceedings/2017/0343.pdf) (IJCAI 2017)
* PTUPCDR: [Personalized Transfer of User Preferences for Cross-domain Recommendation](https://dl.acm.org/doi/pdf/10.1145/3488560.3498392?casa_token=fMj33BdRcdoAAAAA:7iA-ORhh02jV0wY2bPg3keZVcDxAXt5q8hM-9JM8oKrTFj7caBd-HUOICs6gfrIV6tch8NpcYYOC) (WSDM 2022)



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
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

`Processed Dataset`: Fortunately, I have found the processed datasets, and it is convenient to reproduce our results. [Google Drive](https://drive.google.com/file/d/1i-tTB3ffwsR31m1F7F7u-qnTbU63Pl35/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1G9Q86LnkN_XbDAn7pUYHvA?pwd=42ap)

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
Zhu, Yongchun, et al. "Personalized Transfer of User Preferences for Cross-domain Recommendation." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.
```

or in bibtex style:

```
@inproceedings{zhu2022personalized,
  title={Personalized Transfer of User Preferences for Cross-domain Recommendation},
  author={Zhu, Yongchun and Tang, Zhenwei and Liu, Yudan and Zhuang, Fuzhen and Xie, Ruobing and Zhang, Xu and Lin, Leyu and He, Qing},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={1507--1515},
  year={2022}
}
```