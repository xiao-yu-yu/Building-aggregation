"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

import geopandas as gpd
import  pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset
from train import train_epoch, evaluate_network # import train functions




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):

    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name


    root_log_dir, root_ckpt_dir, write_file_name, write_config_file,write_edge_name = dirs
    device = net_params['device']


    train_masks = dataset.train_masks
    val_masks = dataset.val_masks
    test_masks = dataset.test_mask.to(device)
    labels = dataset.labels.to(device)

    # Write network and optimization hyperparameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format
            (DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))


    # At any point you can hit Ctrl + C to break out of training early.
    graph = dataset.g.to(device)
    print("Total num nodes: ", graph.number_of_nodes())
    print("Total num edges: ", graph.number_of_edges())
    node_feat = graph.ndata['feat'].to(device)
    edge_feat = graph.edata['feat'].long().to(device)

    log_dir = os.path.join(root_log_dir, "RUN")
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    same_seeds(params['seed'])

    print("Training Nodes: ", train_masks.int().sum().item())
    print("Validation Nodes: ", val_masks.int().sum().item())
    print("Test Nodes: ", test_masks.int().sum().item())
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    out_epoch =params['epochs']
    output_button=False
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs, epoch_test_accs = [], [], []


    with tqdm(range(params['epochs'])) as t:
        for epoch in t:

            t.set_description('Epoch %d' % epoch)

            start = time.time()

            epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, graph, node_feat,
                                                                       edge_feat, train_masks, labels, epoch)

            epoch_val_loss, epoch_val_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat,
                                                             val_masks, labels, epoch,write_edge_name,output_button=False)
            _, epoch_test_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_masks,
                                                 labels, epoch,write_edge_name,output_button=False)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_accs.append(epoch_train_acc)
            epoch_val_accs.append(epoch_val_acc)
            epoch_test_accs.append(epoch_test_acc)

            writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            writer.add_scalar('train/_acc', epoch_train_acc, epoch)
            writer.add_scalar('val/_acc', epoch_val_acc, epoch)
            writer.add_scalar('test/_acc', epoch_test_acc, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                          train_acc=epoch_train_acc.item(), val_acc=epoch_val_acc.item(),
                          test_acc=epoch_test_acc.item())

            per_epoch_time.append(time.time() - start)

            # Saving checkpoint
            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

            files = glob.glob(ckpt_dir + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch - 1:
                    os.remove(file)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                break

    #         # Stop training after params['max_time'] hours
    #         if time.time() > params[
    #             'max_time'] * 3600 / 20:  # Dividing max_time by 20, since there are 20 splits in WikiCS
    #             print('-' * 89)
    #             print("Max_time for training one split elapsed {:.2f} hours, so stopping".format(params['max_time']))
    #             break
    #
    _, test_acc,result,pred_pro = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, test_masks, labels, epoch,write_edge_name,output_button=True)
    _, valid_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, val_masks, labels,epoch, write_edge_name, output_button=False)
    _, train_acc = evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, train_masks, labels, epoch,write_edge_name,output_button=False)



    #Visualize shapefiles
    label=labels.detach().cpu().numpy()
    mask = torch.nonzero(test_masks).squeeze().cpu().numpy()
    pred_edge_txt = open(write_edge_name, "w")
    for i in range(len(mask)):
        pred_edge_txt.write(str(mask[i]) + "," + str(result[mask[i]]) + "," + str(label[mask[i]]) + '\n')
    pred_edge_txt.close()

    # 过程可视化
    # predictions_array = np.array(pred_pro.cpu())
    #
    # def softmax(x):
    #     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    #     return e_x / e_x.sum(axis=1, keepdims=True)
    #
    #
    # # 计算 Softmax 并输出为数组
    # softmax_predictions = softmax(predictions_array)
    # softmax_array = np.array(softmax_predictions)
    # label=labels.detach().cpu().numpy()
    # mask = torch.nonzero(test_masks).squeeze().cpu().numpy()
    # pred_edge_txt = open(write_edge_name, "w")
    # for i in range(len(mask)):
    #     pred_edge_txt.write(str(mask[i]) + "," + str(result[mask[i]]) + "," + str(label[mask[i]])+ "," + str(softmax_array[mask[i]][1]) + '\n')
    # pred_edge_txt.close()



    test_acc=test_acc.cpu()
    train_acc=train_acc.cpu()
    valid_acc=valid_acc.cpu()


    train_acc_array = [acc.cpu().float().item() for acc in epoch_train_accs]
    train_loss_array = [loss for loss in epoch_train_losses]
    valid_acc_array = [acc.cpu().float().item() for acc in epoch_val_accs]
    valid_loss_array = [loss for loss in epoch_val_losses]

    # 创建两个包含数据的字典，分别用于训练集和验证集
    train_data = {'Train Loss': train_loss_array, 'Train Accuracy': train_acc_array}
    val_data = {'Validation Loss': valid_loss_array, 'Validation Accuracy': valid_acc_array}

    # 将字典转换为DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # 指定要保存的Excel文件名
    excel_file = 'GCN.xlsx'

    # 创建一个Excel Writer 对象
    with pd.ExcelWriter(excel_file) as writers:
        # 将训练集DataFrame写入Excel的第一个工作表
        train_df.to_excel(writers, sheet_name='Train Results', index=False)

        # 将验证集DataFrame写入Excel的第二个工作表
        val_df.to_excel(writers, sheet_name='Validation Results', index=False)

    print(f'实验结果已保存到 {excel_file}')

    print("Valid Accuracy [LAST EPOCH]: {:.4f}".format(valid_acc))
    print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
    print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
    print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))

    writer.close()


    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time( ) -t0 ) /3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))


    """
    #     Write the results in out_dir/results folder
    # """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n
        Total Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        (time.time( ) -t0 ) /3600, np.mean(per_epoch_time)))






def main():
    """
        USER CONTROLS
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    # parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    # parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    # parser.add_argument('--readout', help="Please give a value for readout")
    # parser.add_argument('--kernel', help="Please give a value for kernel")
    # parser.add_argument('--gated', help="Please give a value for gated")
    # parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    # if args.edge_feat is not None:
    #     net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator

    net_params['in_dim'] = dataset.n_feats
    net_params['n_classes'] = dataset.num_classes
    net_params['e_features'] = dataset.e_feats

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_edge_file = out_dir + 'edge/edge_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str \
        (config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y.txt')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file,write_edge_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    if not os.path.exists(out_dir + 'edge'):
        os.makedirs(out_dir + 'edge')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)







main()