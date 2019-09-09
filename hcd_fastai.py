import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import error_rate
from torchvision.models import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 2019
path = '../input/histopathologic-cancer-detection/'
output_path = './'

parser = argparse.ArgumentParser()
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--img-size', type=int, default=196)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs')
args = parser.parse_args()


def kfold_wsi(nb_folds, wsi_df, missing_df, random_state=2019):
    ## cross validation split by wsi id
    np.random.seed(random_state)

    # wsi attributions
    wsi_id, wsis = wsi_df['wsi'], wsi_df['wsi'].unique()
    wsi_count = wsi_df.wsi.value_counts()
    wsi_count.name = 'count'
    wsi_mean = wsi_df.groupby('wsi')['label'].mean()
    wsi_mean.name = 'label_mean'
    wsi_attr = pd.concat([wsi_count, wsi_mean], axis=1)
    wsi_attr.sort_values('count', axis=0, ascending=False, inplace=True)

    # stratified split for cv
    sample_per_fold = len(wsi_id) // nb_folds
    val_wsis = []
    for i in range(nb_folds-1):
        # create valid set wsi id
        print("collecting set %d..." %(i+1))
        val_wsi_id2cnt = {}
        nb_val = 0
        index_to_select = wsi_attr.loc[wsis][wsi_attr.loc[wsis]['label_mean']>=0.5].index
        while nb_val < sample_per_fold * 0.4:
            w_id = np.random.choice(index_to_select,1)[0]
            if w_id not in val_wsi_id2cnt:
                val_wsi_id2cnt[w_id] = wsi_attr.loc[w_id]['count']
                nb_val += val_wsi_id2cnt[w_id]
        index_to_select = wsi_attr.loc[wsis][wsi_attr.loc[wsis]['label_mean']<0.5].index
        while nb_val < sample_per_fold:
            w_id = np.random.choice(index_to_select,1)[0]
            if w_id not in val_wsi_id2cnt:
                val_wsi_id2cnt[w_id] = wsi_attr.loc[w_id]['count']
                nb_val += val_wsi_id2cnt[w_id]
        val_wsis.append(list(val_wsi_id2cnt.keys()))
        wsis = list(set(wsis) - set(val_wsis[i]))
    print("collecting set {}...".format(nb_folds))
    val_wsis.append(wsis)

    # handling missing wsi_id image indices
    missing_indices = np.array(missing_df.index)
    skf = StratifiedKFold(n_splits=nb_folds, random_state=random_state, shuffle=False)
    split_idxs = [(tr_idx, val_idx)
                  for tr_idx, val_idx in skf.split(np.zeros(missing_df.shape[0]), missing_df['label'].values)]
    ms_trn_indices = [missing_indices[split_idxs[i][0]] for i in range(nb_folds)]
    ms_val_indices = [missing_indices[split_idxs[i][1]] for i in range(nb_folds)]

    # generate trn/val indices
    train_indices = []
    valid_indices = []
    for i in range(nb_folds):
        valid_indices.append( np.where(wsi_id.isin(val_wsis[i]))[0] )
        tr_idx = np.array([])
        for j in set(range(nb_folds))-{i}:
            tr_idx = np.concatenate([ tr_idx, np.where(wsi_id.isin(val_wsis[j]))[0] ])
        tr_idx.sort()
        train_indices.append(tr_idx.astype('int'))

    # combine 2 set of indices
    train_indices = [np.concatenate([tr_idxs, ms_tr_idxs]) for tr_idxs, ms_tr_idxs in zip(train_indices, ms_trn_indices)]
    valid_indices = [np.concatenate([val_idxs, ms_val_idxs]) for val_idxs, ms_val_idxs in zip(valid_indices, ms_val_indices)]

    return train_indices, valid_indices


def load_meta_data():
    train_df = pd.read_csv(path+'train_labels.csv')
    wsi_df = pd.read_csv('../input/wsi-id/patch_id_wsi.csv')

    df = train_df.merge(wsi_df, how='outer', on='id')
    wsi_df = df[~df.wsi.isna()]
    missing_df = df[df.wsi.isna()]

    return train_df, wsi_df, missing_df


def main(args):
    # load meta data
    train_df, wsi_df, missing_df = load_meta_data()

    # training preparation
    train_targets = np.zeros(train_df.shape[0], dtype='int32')
    train_preds = np.zeros(train_df.shape[0], dtype='float32') # matrix for the out-of-fold predictions
    test_preds = np.zeros(57458, dtype='float32') # matrix for the predictions on the testset
    cv_indices = kfold_wsi(args.n_splits, wsi_df, missing_df)

    # start training
    print()
    for i, (trn_idx, val_idx) in enumerate(zip(*cv_indices)):
        print(f'Fold {i + 1}')

        # prepare data
        src = (ImageList.from_df(df=train_df, path=path, folder='train', suffix='.tif')
               .split_by_idxs(trn_idx, val_idx)
               .label_from_df(cols='label'))

        data = (src.transform(tfms=get_transforms(flip_vert=True), size=args.img_size)
                .add_test_folder('test')
                .databunch(bs=args.batch_size).normalize(imagenet_stats))

        # prepare model
        learn = cnn_learner(data, models.resnet50, metrics=[accuracy])
        learn.model_dir = output_path

        # train
        learn.fit_one_cycle(args.epochs, max_lr=args.lr, callbacks=[SaveModelCallback(learn, name=f'rn-50-sz196-{i+1}')])

        # inference
        pred_val, y_val = learn.get_preds(ds_type=DatasetType.Valid)
        train_targets[val_idx] = y_val.numpy()
        train_preds[val_idx] = pred_val.numpy()[:,1]

        pred_test, _ = learn.get_preds(ds_type=DatasetType.Test)
        test_preds += pred_test.numpy()[:,1] / args.n_splits

        print()

    # make submission
    print(f'val auc cv score is {roc_auc_score(train_targets, train_preds)}')
    submit = pd.DataFrame({'id':[f.split('.')[0] for f in os.listdir(path+'test')], 'label':test_preds})
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)