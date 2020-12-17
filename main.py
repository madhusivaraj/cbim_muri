import os
import time 
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import average_precision_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.tcn as tcn
import models.rnns as rnns
from dataset import CRP
from utils import AverageMeter, compute_accuracy


#####
## main
# 1. arguments
# 2. paths to save files
# 3. model
# 4. criterion
# 5. optimizer
# 6. data augmentation
# 7. data loader
## train
## validate
#####

    
## arguments 
parser = argparse.ArgumentParser()

# model params
parser.add_argument('--model',       default='LSTM',        type=str,   help='name of the model to use', choices=['GRU', 'LSTM', 'TCN'])
parser.add_argument('--aggregation', default='average',     type=str,   help='name of the model to use', choices=['last', 'average', 'max', 'attention'])
parser.add_argument('--temp_stride', default=5,             type=int,   help='number of temporal stride to use')
parser.add_argument('--in_features', default=17,            type=int,   help='number of input features/channels')
parser.add_argument('--num_classes', default=2,             type=int,   help='number of classes to use for classification')
# params only for LSTM/GRU
parser.add_argument('--hidden_size', default=128,           type=int,   help='hidden state dimension for LSTM')
parser.add_argument('--num_layers',  default=2,             type=int,   help='number of layers to use')
# params only for TCN
parser.add_argument('--num_channels', default='64,64,128',   type=str,   help='list with channel numbers for TCN')
parser.add_argument('--kernel_size',  default=3,             type=int,   help='kernel size for TCN')
parser.add_argument('--dropout',      default=0.1,           type=float, help='dropout rate')
# --------------
# data params
parser.add_argument('--root',        default='data',        type=str,    help='path to root of the dataset')
parser.add_argument('--save_dir',    default='exps/chpts',  type=str,    help='path to save models')
parser.add_argument('--log_dir',     default='exps/logs',   type=str,    help='path used for logging')
parser.add_argument('--tb_dir',      default='exps/tb',     type=str,    help='path to store tensorboards')
parser.add_argument('--debug',       default=False, action='store_true', help='use this flag in debug mode')
# training procedure params
parser.add_argument('--start-epoch', default=0,             type=int,    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs',      default=30,            type=int,    help='number of total epochs to run')
parser.add_argument('--print_freq',  default=5,             type=int,    help='frequency of printing output during training')
parser.add_argument('--batch_size',  default=256,           type=int,    help='number of sample per batch')
parser.add_argument('--num_workers', default=4,             type=int,    help='number of subprocesses to use for data loading')
# hyperparameters
parser.add_argument('--lr',          default=1e-4,          type=float,  help='learning rate')
parser.add_argument('--wd',          default=1e-5,          type=float, help='weight decay')
parser.add_argument('--seed',        default=14,            type=int,    help='seed to be used during random generations for reproducability')

# 1. parse arguments
args   = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
args.num_channels = list( map(int, args.num_channels.split(',')) )
if args.debug:
    args.batch_size  = 4
    args.num_workers = 0
    args.epochs      = 5
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True


# ----------

def get_dataloader(mode='train', transform=None):
    dataset = CRP(args, mode=mode, transform=transform)
    if mode == 'train':
        data_loader = DataLoader(dataset    = dataset,
                                batch_size  = args.batch_size,
                                shuffle     = True,
                                num_workers = args.num_workers,
                                pin_memory  = True,
                                drop_last   = False
                                )
    elif mode == 'val':
        data_loader = DataLoader(dataset    = dataset,
                                # batch_size  = args.batch_size,
                                batch_size  = 313,
                                shuffle     = False,
                                num_workers = args.num_workers,
                                pin_memory  = True,
                                drop_last   = False
                                )
    elif mode == 'test':
        data_loader = DataLoader(dataset    = dataset,
                                # batch_size  = args.batch_size,
                                batch_size  = 583,
                                shuffle     = False,
                                num_workers = args.num_workers,
                                pin_memory  = True,
                                drop_last   = False
                                )
    logging.info('=> Loading {} samples for {}'.format(len(dataset), mode))
    return data_loader


# ----------

def main():
    start_time = time.time()

    # 2. create directories to save experiment results 
    exp_name      =  '{}_layers{}_stride{}_{}_lr{}'.format(args.model, args.num_layers, args.temp_stride, args.aggregation, args.lr)
    args.save_dir = os.path.join(args.save_dir, exp_name) 
    if not os.path.exists(args.save_dir):
    	os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not args.debug:
        args.tb_dir   = os.path.join(args.tb_dir, exp_name)
        if not os.path.exists(args.tb_dir):
        	os.makedirs(args.tb_dir)
        writer_val   = SummaryWriter(log_dir=os.path.join(args.tb_dir, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(args.tb_dir, 'train'))
    

    # create logger
    logging.basicConfig(filename=os.path.join(args.log_dir, exp_name)+'.log', level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    console.setLevel(logging.INFO)

    logging.info(args)

    # 3. create model
    if args.model == 'GRU':
        model = rnns.GRU(args.in_features, args.hidden_size, args.num_layers, args.aggregation, args.num_classes, args.device)
    elif args.model == 'LSTM':
        model = rnns.LSTM(args.in_features, args.hidden_size, args.num_layers, args.aggregation, args.num_classes, args.device)
    elif args.model == 'TCN':
        model = tcn.TemporalConvNet(in_channels=args.in_features, num_channels=args.num_channels, kernel_size=args.kernel_size,
                                            aggregation=args.aggregation, num_classes=args.num_classes, dropout=args.dropout)
    else:
    	raise ValueError('Wrong model!')

    model = model.to(args.device)

    # log the model
    logging.info(model)

    # 4. define criterion
    # W_tensor  = torch.Tensor([1, 1.6]).to(args.device)
    W_tensor = torch.Tensor([1, 2]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=W_tensor)

    # 5. define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6. define data loader
    train_loader = get_dataloader('train')
    val_loader   = get_dataloader('val'  )
    test_loader  = get_dataloader('test' )


    ## Main Loop
    best_tp, best_ap, best_epoch = 0, 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_ap, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        val_loss, val_ap, val_acc , val_tp = validate(val_loader, model, criterion, epoch)

        if not args.debug:
            # save curve
            writer_train.add_scalar('loss',             train_loss, epoch)
            writer_train.add_scalar('average-precision', train_ap,  epoch)
            writer_val.add_scalar('loss',               val_loss,   epoch)
            writer_val.add_scalar('average-precision',   val_ap,    epoch)

        # save check_point
        is_best = val_ap > best_ap
        best_tp = max(val_tp, best_tp)
        best_ap = max(val_ap, best_ap)
        # save the best model
        if is_best:
            best_epoch = epoch
            if not args.debug:
                logging.info('=> AP improved ({:.2f}) improved at epoch {} | saving best model!'.format(100*val_ap, epoch))
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth') )

    logging.info('### Training from epoch {} -> {} finished in ({:.2f}) minutes'.format(args.start_epoch, args.epochs-1, (time.time()-start_time)/60 )  ) 
    logging.info('### Best validation AP: {:.2f} in epoch {}'.format(100*best_ap, best_epoch) )
    if not args.debug:
        test(test_loader, model, exp_name, criterion)


def train(data_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    AP     = AverageMeter()
    ACC    = AverageMeter()
    TP     = AverageMeter()
    FP     = AverageMeter()
    FN     = AverageMeter()
    TN     = AverageMeter()

    model.train()
    for idx, data in enumerate(data_loader):
        inputs, labels = data
        B 			   = inputs.size(0)
        if torch.cuda.is_available():
            inputs = inputs.to(args.device, non_blocking=True)  # GRU/LSTM : [B, T, C] | TCN: [B, C, T]
            labels = labels.to(args.device, non_blocking=True)  # [B]


        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)               # [B, 2]
        
        # update statistics
        loss     = criterion(outputs, labels)
        scores   = F.softmax(outputs, dim=1)  # [B, 2]
        _, preds = torch.max(outputs, dim=1)  # [B]
        
        acc    = compute_accuracy(scores, labels)
        labels = labels.detach().cpu().numpy()
        ap     = average_precision_score(y_true=labels, y_score=scores[:,1].detach().cpu().numpy() )
        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds.detach().cpu().numpy()).ravel()
        
        TP.update(tp, B)
        FP.update(fp, B)
        FN.update(fn, B)
        TN.update(tn, B)
        AP.update(ap, B)
        ACC.update(acc, B)
        losses.update(loss.item(),  B)

        # backward pass + parameter update
        loss.backward()
        optimizer.step()

    ap_print  = 100*AP.avg
    acc_print = 100*ACC.avg
    logging.info('TRAIN | Epoch: [{0}/{1}] | Loss: {loss.avg:.4f} | AP: {ap:.2f} | TP:{tp.avg:.2f}, FP:{fp.avg:.2f}, FN:{fn.avg:.2f}, TN:{tn.avg:.2f} | ACC: {acc:.2f}'.
                                                                format(epoch, args.epochs, ap=ap_print, tp=TP, fp=FP, fn=FN, tn=TN, acc=acc_print, loss=losses))

    return losses.avg, AP.avg, ACC.avg


def validate(data_loader, model, criterion, epoch):
    losses = AverageMeter()
    AP     = AverageMeter()
    ACC    = AverageMeter()
    TP     = AverageMeter()
    FP     = AverageMeter()
    FN     = AverageMeter()
    TN     = AverageMeter()

    model.eval()
    with torch.no_grad():
    	for idx, data in enumerate(data_loader):
            inputs, labels = data
            B              = inputs.size(0)
            if torch.cuda.is_available():
            	inputs = inputs.to(args.device, non_blocking=True)  # GRU/LSTM : [B, T, C] | TCN: [B, C, T]
            	labels = labels.to(args.device, non_blocking=True)  # [B]

            # forward pass
            outputs = model(inputs) 			  # [B, 2]

            # update statistics
            loss     = criterion(outputs, labels)
            scores   = F.softmax(outputs, dim=1)  # [B, 2]
            _, preds = torch.max(outputs, dim=1)  # [B]

            acc    = compute_accuracy(scores, labels)
            labels = labels.detach().cpu().numpy()
            ap     = average_precision_score(y_true=labels, y_score=scores[:,1].detach().cpu().numpy() )
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds.detach().cpu().numpy()).ravel()

            TP.update(tp, B)
            FP.update(fp, B)
            FN.update(fn, B)
            TN.update(tn, B)
            AP.update(ap, B)
            ACC.update(acc, B)
            losses.update(loss.item(),  B)

    ap_print  = 100*AP.avg
    acc_print = 100*ACC.avg
    logging.info('VAL   | Epoch: [{0}/{1}] | Loss: {loss.avg:.4f} | AP: {ap:.2f} | TP:{tp.avg}, FP:{fp.avg}, FN:{fn.avg}, TN:{tn.avg} | ACC: {acc:.2f}'.
                                                                format(epoch, args.epochs, ap=ap_print, tp=TP, fp=FP, fn=FN, tn=TN, acc=acc_print, loss=losses))

    return losses.avg, AP.avg, ACC.avg, tp


def test(data_loader, model, exp_name, criterion):
    losses = AverageMeter()
    AP     = AverageMeter()
    ACC    = AverageMeter()
    TP     = AverageMeter()
    FP     = AverageMeter()
    FN     = AverageMeter()
    TN     = AverageMeter()

    print("=> Loading model for testing!")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    model.eval()
    print("=> Model is ready!")
    with torch.no_grad():
    	for idx, data in enumerate(data_loader):
            inputs, labels = data
            B              = inputs.size(0)
            if torch.cuda.is_available():
            	inputs = inputs.to(args.device, non_blocking=True)  # GRU/LSTM : [B, T, C] | TCN: [B, C, T]
            	labels = labels.to(args.device, non_blocking=True)  # [B]
                
            # forward pass
            outputs = model(inputs)               # [B, 2]

            # update statistics
            loss     = criterion(outputs, labels)
            scores   = F.softmax(outputs, dim=1)  # [B, 2]
            _, preds = torch.max(outputs, dim=1)  # [B]

            acc    = compute_accuracy(scores, labels)
            labels = labels.detach().cpu().numpy()
            ap     = average_precision_score( y_true=labels, y_score=scores[:,1].detach().cpu().numpy() )
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds.detach().cpu().numpy()).ravel()

            TP.update(tp, B)
            FP.update(fp, B)
            FN.update(fn, B)
            TN.update(tn, B)
            AP.update(ap, B)
            ACC.update(acc, B)
            losses.update(loss.item(),  B)

    ap_print  = 100*AP.avg
    acc_print = 100*ACC.avg
    logging.info('# Test Set | Loss: {loss.avg:.4f} | AP: {ap:.2f} | TP:{tp.avg}, FP:{fp.avg}, FN:{fn.avg}, TN:{tn.avg} | ACC: {acc:.2f}'.
                                                                        format(ap=ap_print, tp=TP, fp=FP, fn=FN, tn=TN, acc=acc_print, loss=losses))
    return

if __name__ == '__main__':
    main()
