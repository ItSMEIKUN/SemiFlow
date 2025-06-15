import argparse
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from models.DCC import DCC
from models.SupConLoss import SupConLoss
from tools import data_processor
from tools.data_split import split_data
from tools.evaluator import measurement
from tools.methods import *

threshold = 0.95


def unlabel_args(params):
    dataset = params.dataset
    setting = params.setting
    device = torch.device(params.device)
    num_workers = params.num_workers
    batch_size = params.batch_size
    n_label = params.n_label
    n_ow = params.n_ow

    learning_rate = 5e-4
    in_path = os.path.join('datasets', dataset, 'CW.npz')
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"The datasets path does not exist: {in_path}")
    X, y = data_processor.load_data(in_path)
    num_classes = len(np.unique(y))

    # Split data sets
    train_X, train_y, test_X, test_y, train_ulabel = split_data(X, y, train_size=n_label * num_classes,
                                                                test_size=200 * num_classes,
                                                                unlabel_size=50 * num_classes)
    unlabel_1 = train_ulabel
    unlabel_2 = train_ulabel
    unlabel_s = augmented_data(train_ulabel)
    if setting == 'OW':
        ow_data = np.load(os.path.join( 'datasets', dataset, 'OW.npz'), allow_pickle=True)
        X = ow_data['X']
        train_size = len(train_X)
        test_size = n_ow * len(test_X)
        ulabel_size = len(train_ulabel)
        indices = np.random.choice(len(X), size=(train_size + test_size + ulabel_size), replace=False)
        # Extract the corresponding samples and labels
        X = X[indices]
        y = np.full(shape=(train_size + test_size + ulabel_size), fill_value=(train_y.max() + 1))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        ow_train_x, ow_unlabel_x, ow_test_x = divide_array_by_sizes(X, train_size, ulabel_size, test_size)
        ow_train_y, _, ow_test_y = divide_array_by_sizes(y, train_size, ulabel_size, test_size)
        train_X, train_y = torch.cat((train_X, ow_train_x)), torch.cat((train_y, ow_train_y))
        test_X, test_y = torch.cat((test_X, ow_test_x)), torch.cat((test_y, ow_test_y))
        unlabel_1 = torch.cat((unlabel_1, ow_unlabel_x))
        unlabel_2 = torch.cat((unlabel_2, ow_unlabel_x))
        unlabel_s = torch.cat((unlabel_s, augmented_data(ow_unlabel_x)))
    unlabeled_data = TensorDataset(unlabel_1, unlabel_2, unlabel_s)
    num_classes = len(np.unique(train_y))
    assert num_classes == train_y.max() + 1, "Labels are not continuous"
    print(f"Train: X={train_X.shape}, y={train_y.shape}")
    print(f"Test: X={test_X.shape}, y={test_y.shape}")
    print(f"Ulabel: X={unlabel_1.shape}")
    print(f"num_classes: {num_classes}")

    train_iter = data_processor.load_iter(train_X, train_y, batch_size, True, num_workers)
    test_iter = data_processor.load_iter(test_X, test_y, 10 * batch_size, True, num_workers)
    train_ulabel_iter = data_processor.load_iter(unlabeled_data,None, 2 * batch_size, True, num_workers)
    model = DCC(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    return model, optimizer, train_iter, test_iter, train_ulabel_iter, num_classes, device, setting


def get_label(model, u1, u2, s,device):
    all_predictions = []
    with torch.no_grad():
        for i in range(5):
            out_w, _ = model(u1.to(device))
            out_w = F.softmax(out_w, dim=1)
            all_predictions.append(out_w)

    # Stacking and averaging all projections
    dropout_predictions = torch.stack(all_predictions, dim=1)
    averaged_predictions = torch.mean(dropout_predictions, dim=1)

    # Get the maximum probability value and its corresponding category
    max_probs, max_idx = torch.max(averaged_predictions, dim=1)

    # Masks for generating pseudo-labels via thresholding
    mask = max_probs.ge(threshold).float()

    # Pseudo-labeling
    label_w = max_idx[mask == 1].long()

    mask = mask.cpu()
    sup_u1 = u1[mask == 0]
    sup_u2 = u2[mask == 0]

    # Strong enhancement operation and selection of plausible unlabeled data by mask
    u_s = s[mask == 1]

    return sup_u1, sup_u2, u_s, label_w


def unlabel_train(
        model,
        optimizer,
        train_iter,
        test_iter,
        train_ulabel_iter,
        train_epochs,
        num_classes,
        device,
        setting):
    best_acc = 0
    CELoss = torch.nn.CrossEntropyLoss()
    SupLoss = SupConLoss()

    for epoch in range(train_epochs):
        labeled_iter = iter(train_iter)
        model.train()
        total_loss_x, total_loss_u, total_loss_s = 0, 0, 0
        for _, (u1, u2, s) in enumerate(train_ulabel_iter):
            try:
                inputs_x, label_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_iter)
                inputs_x, label_x = next(labeled_iter)
            label_size = inputs_x.size(0)
            sup_u1, sup_u2, u_s, label_w = get_label(model, u1, u2, s,device)
            inputs = torch.cat((inputs_x, sup_u1, sup_u2, u_s))
            optimizer.zero_grad()

            outputs, features = model(inputs.to(device))
            outputs_x = outputs[:label_size]
            if u_s.size(0) != 0:
                outputs_s = outputs[-u_s.size(0):]
                loss_u = CELoss(outputs_s, label_w)
            else:
                loss_u = 0
            features_w1, features_w2 = torch.chunk(features[label_size:features.size(0) - u_s.size(0)], 2)
            loss_x = CELoss(outputs_x, label_x.to(device))
            loss_s = SupLoss(features_w1, features_w2, features_w1.shape[0]) if features_w1.shape[
                                                                                    0] > 0 else torch.tensor(0.0,
                                                                                                             device=device)
            loss = loss_x + loss_u + 0.1 * loss_s
            loss.backward()
            optimizer.step()
            total_loss_x += loss_x
            total_loss_u += loss_u
            total_loss_s += loss_s
        print(
            f"epoch {epoch + 1}:loss_x={total_loss_x}, loss_u={total_loss_u}, loss_s={total_loss_s}\n ")
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                valid_pred = []
                valid_true = []
                ow_label = num_classes - 1
                for _, cur_data in enumerate(test_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                    outs, _ = model(cur_X)
                    # Convert the output to a probability distribution using the softmax function
                    outs = torch.softmax(outs, dim=1)
                    valid_pred.append(outs.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())
                valid_pred = np.concatenate(valid_pred)
                valid_true = np.concatenate(valid_true)
                if setting == 'OW':
                    thresholds = np.arange(0.1, 1, 0.1)
                    for th in thresholds:
                        print(f'--------------------- threshold = {th:.1f}')
                        valid_result = measurement(valid_true, valid_pred, ow_label, th)
                        print(f"{valid_result}")
                else:
                    valid_result = measurement(valid_true, valid_pred, ow_label, 0)
                    if valid_result["Accuracy"] > best_acc:
                        best_acc = valid_result["Accuracy"]
                    print(f"{epoch + 1}: {valid_result}")
    print(f"Best Acc: {best_acc}")


if __name__ == '__main__':
    # Argument parser for command-line options, arguments, and sub-commands
    # Input parameters
    parser = argparse.ArgumentParser(description="SemiFlow")
    parser.add_argument("--dataset", type=str, default="AWF", help="Dataset name")
    parser.add_argument("--setting", type=str, default="CW", help="setting")
    parser.add_argument("--device", type=str, default="cuda", help="Device, options=[cpu, cuda, cuda:x]")

    # Optimization parameters
    parser.add_argument("--num_workers", type=int, default=5, help="Data loader num workers")
    parser.add_argument("--train_epochs", type=int, default=400, help="Train epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size of train input data")

    # Other Setting Parameters
    parser.add_argument("--n_label", type=int, default=5, help="Number of labels per website")
    parser.add_argument("--n_ow", type=int, default=1, help="Open-world datasets of different quantitative sizes")
    params = parser.parse_args()
    model, optimizer, train_iter, test_iter, train_ulabel_iter, num_classes, device, setting = unlabel_args(params)
    unlabel_train(
        model,
        optimizer,
        train_iter,
        test_iter,
        train_ulabel_iter,
        params.train_epochs,
        num_classes,
        device,
        setting)
