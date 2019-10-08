import time
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
model_path = '../models/'
output_path = './'

# Learning parameters
batch_size = 128  # batch size
img_size = 196    # resized image size
n_splits = 5      # K-fold data split
which_fold = 0    # should be int in [0, n_splits-1]
tta = True        # test time augmentation
is_nnavg = True   # ensemble models from weight averaging or not


def load_meta_data():
    return pd.read_csv(data_path+'train_labels.csv')

def main():
    test_id = [f.split('.')[0] for f in os.listdir(data_path+'test')]

    # Load pretrained net
    net = models.resnet50(pretrained=True)

    # Prepare data
    if tta:
        test_datasets = [HCDDataset(path)] + [HCDDataset(path, tta=True)]*6
    else:
        test_dataset = HCDDataset(path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if not is_nnavg:
        model_states = torch.load(model_path+'models.pt')
        model = HCDNet(copy.deepcopy(net))
        model.load_state_dict(model_states[f'fold_{which_fold}'])

        model = model.to(device)
        if tta:
            pred_test = TTA(test_datasets, model)
        else:
            pred_test = get_preds(test_loader, model)

    else:
        train_df = load_meta_data()
        mask = np.random.rand(len(train_df)) < 0.8
        trn_idx = np.arange(len(train_df))[mask]
        train_dataset = HCDDataset(data_path, trn_idx, df=train_df)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        avgd_model = HCDNet(copy.deepcopy(net))
        avgd = NNAverage(avgd_model, 1./n_splits)

        for i in range(n_splits):
            m_path = model_path+f'hcd-rn50-{i}/'
            model_states = torch.load(m_path+'models.pt')
            model = HCDNet(copy.deepcopy(net))
            model.load_state_dict(model_states[f'fold_{i}'])
            avgd.update(model)

        avgd.set_weights(avgd_model)
        avgd_model = avgd_model.to(device)
        forward_model(avgd_model, train_loader)

        if tta:
            pred_test = TTA(test_datasets, avgd_model)
        else:
            pred_test = get_preds(test_loader, avgd_model)


    submit = pd.DataFrame({'id':test_id, 'label':pred_test})
    submit.to_csv(output_path+'submission.csv', index=False)


if __name__ == '__main__':
    main()
