import time
import torch.backends.cudnn as cudnn
import torch
from torchvision import models
from torch.utils.data import DataLoader
from datasets import *
from model import *
from solver import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2019

# Data parameters
data_path = '../input/'  # path of data files
output_path = './'

# Learning parameters
batch_size = 96  # batch size
epochs = 4  # number of epochs to run
lr = 1e-3  # learning rate
img_size = 196  # resized image size
n_splits = 5
which_fold = 0  # should be int in [0, n_splits-1]
ckpt_per_fold = True
enable_ckpt_ensemble = False
keep_ema = False

# grad_clip = None  # Use a value of 0.5 if gradients are exploding, which may happen
                  # at larger batch sizes (sometimes at 32) - you will recognize it
                  # by a sorting error in the MuliBox loss calculation

# cudnn.benchmark = True


def main():
    """
    Training and validation.
    """

    # Data stuffs
    train_df = load_meta_data()
    # nb_train = train_df.shape[0]  # 220025
    # nb_test = len(os.listdir(data_path+'test'))  # 57458

    # training preparation
    fold_val_preds = []     # oof predictions for ensemble of folds
    ckpt_val_preds = []     # oof predictions for ensemble of ckpts
    ema_val_preds = []      # oof predictions for ensemble from ema of weights
    oof_targets = []
    cv_indices = train_val_split(train_df, which_fold)
    # test_dataset = HCDDataset(data_path)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load pretrained net
    net = models.resnet50(pretrained=True)

    # Training
    model_ckpts = {}
    for _, (trn_idx, val_idx) in enumerate(cv_indices):
        i = which_fold
        print(f'Fold {i+1}')

        # prepare data
        train_dataset = HCDDataset(data_path, trn_idx, train_df)
        val_dataset = HCDDataset(data_path, val_idx, train_df)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        oof_targets.append(val_dataset.targets)

        # prepare model
        seed_torch(SEED+i)
        model, optimizer = model_optimizer_init(net)
        ft_lrs = [lr*0.01, lr*0.1, lr]
        scheduler = OneCycleScheduler(optimizer, epochs, train_loader, max_lr=ft_lrs)
        model = model.to(device)

        if keep_ema:
            ema_model = copy.deepcopy(model)
            ema_model.eval()
            ema = WeightEMA(model, sample_rate=50)

        # train
        t_start = time.time()
        single_val_preds = []
        for e in range(epochs):
            print(f'Epoch: {e+1}')

            train_one_cycle(train_loader, model, optimizer, scheduler, e, ema)
            val_scores, val_auc = validate(val_loader, model, e)

            # for ckpt ensemble
            if enable_ckpt_ensemble:
                single_val_preds.append(val_scores)
                model_ckpts[f'fold_{i}_epk_{e}'] = model.state_dict()

            # for best ckpt per fold
            if ckpt_per_fold:
                if e == 0:
                    best_auc = val_auc
                if val_auc > best_auc:
                    print('updating best val auc and model...')
                    best_auc = val_auc
                    model_ckpts[f'fold_{i}'] = model.state_dict()
                    best_val_scores = val_scores

        time_elapsed = time.time() - t_start
        print('Fold {} done in {:.0f}m {:.0f}s.'.format(i+1, time_elapsed // 60, time_elapsed % 60))
        torch.cuda.empty_cache()
        print(str(torch.cuda.memory_allocated(device)/1e6 ) + 'M')
        print(str(torch.cuda.memory_cached(device)/1e6 ) + 'M')

        # inference
        if ckpt_per_fold:
            fold_val_preds.append(best_val_scores)

        if enable_ckpt_ensemble:
            ckpt_weights = [2**e for e in range(epochs)]
            val_scores = np.average(single_val_preds, weights=ckpt_weights, axis=0)
            ckpt_val_preds.append(val_scores)
            print('{"metric": "Ckpt CV Val. AUC", "value": %.4f, "epoch": %d}' % (
                roc_auc_score(val_dataset.targets, val_scores), e+1))

        if keep_ema:
            ema.set_weights(ema_model)
            forward_model(ema_model, train_loader)
            model_ckpts[f'ema_fold_{i}'] = ema_model.state_dict()
            val_scores = get_preds(val_loader, ema_model, 'val')
            ema_val_preds.append(val_scores)
            print('{"metric": "EMA Val. AUC", "value": %.4f, "epoch": %d}' % (
                roc_auc_score(val_dataset.targets, val_scores), e+1))

        print()

    # Show validation results
    oof_targets = np.concatenate(oof_targets)

    if ckpt_per_fold:
        fold_val_preds = np.concatenate(fold_val_preds)
        fold_cv_auc = roc_auc_score(oof_targets, fold_val_preds)
        print(f'For whole oof set, fold cv val auc score: {fold_cv_auc}')

    if enable_ckpt_ensemble:
        ckpt_val_preds = np.concatenate(ckpt_val_preds)
        ckpt_cv_auc = roc_auc_score(oof_targets, ckpt_val_preds)
        print(f'For whole oof set, ckpt cv val auc score: {ckpt_cv_auc}')

    if keep_ema:
        ema_val_preds = np.concatenate(ema_val_preds)
        ema_auc = roc_auc_score(oof_targets, ema_val_preds)
        print(f'For whole oof set, ema val auc score: {ema_auc}')

    # Save model checkpoints
    torch.save(models, output_path+'models.pt')


def load_meta_data():
    return pd.read_csv(data_path+'train_labels.csv')


def train_val_split(train_df, which_fold=None):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in kf.split(train_df['id'], train_df['label'])]
    if which_fold is not None:
        return [cv_indices[which_fold]]
    return cv_indices


if __name__ == '__main__':
    main()
