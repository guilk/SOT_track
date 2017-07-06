import os
import argparse
import numpy as np
from symbols import *
from mxnet.optimizer import SGD
import loss_layers
import logging
from tensorboard import SummaryWriter
import time
import cPickle
from triplet_iterator import triplet_iterator
from mutable_module import MutableModule

from collections import namedtuple
Batch = namedtuple('Batch',['data'])

def get_data_iter():

    tr_data_iter = triplet_iterator(split_type = 'triplet_train', aug_params=args)
    val_data_iter = triplet_iterator(split_type = 'triplet_val', aug_params=args)
    num_labels, num_samples = tr_data_iter.get_stats()
    # return val_data_iter
    return tr_data_iter, val_data_iter, num_labels, num_samples

def parse_args():
    '''
    :return: Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Train SINT using ReID')

    parser.add_argument('--batch_size',dest='batch_size',
                        help='bacth size used for training', default=8, type=int)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='If shuffling data at first epoch',default=False)
    parser.add_argument('--db_root', dest='db_root',
                        help='The root directory of the dataset', default='/data01/ALOV', type=str)
    parser.add_argument('--db_name', dest='db_name',
                        help='The name of the dataset', default='ALOV', type=str)
    parser.add_argument('--imageSz', dest='imageSz',
                        help='The resized image size', default=224, type=int)
    parser.add_argument('--lr', dest = 'lr',
                        help='base learning rate', default=0.001, type = float)
    parser.add_argument('--sampling', dest = 'sampling',
                        help='training data sub-sampling', default=1, type = int)
    parser.add_argument('--rand_mirror', dest='rand_mirror',action='store_true', default=False,
                        help='random mirror image for augmentation')
    parser.add_argument('--lr_step', dest='lr_step',
                        help='learning rate steps (in epoch)', default='To do', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu ids used for training', default='0', type=str)
    parser.add_argument('--num_rois', dest='num_rois',
                        help='num of rois in one image for training', default=16, type=int)
    parser.add_argument('--lsoftmax', dest='lsoftmax', action='store_true', default=False,
                        help='Set true if using lsoftmax')

    parser.add_argument('--verifi', dest='verifi',action='store_true',
                        help='Set true if using verification loss', default=True)
    parser.add_argument('--verifi_threshd', dest='verifi_threshd', type=float, default=1.6,
                        help='the threshold used for verification loss')

    parser.add_argument('--lmnn', dest='lmnn', action='store_true',
                        help='Set true if using lmnn loss', default=False)
    parser.add_argument('--lmnn_threshd', dest='lmnn_threshd', type=float, default=0.9,
                        help='the threshold used for lmnn loss')
    parser.add_argument('--lmnn_epsilon', dest='lmnn_epsilon', type=float, default=0.1,
                        help='the epsilon used for lmnn loss')

    parser.add_argument('--triplet', action='store_true', default=False,
                        help='if use triplet loss')
    parser.add_argument('--triplet-weight', type=float, default=1.0,
                        help='triplet loss weight')
    parser.add_argument('--triplet-threshd', type=float, default=0.9,
                        help='triplet threshold')

    parser.add_argument('--mode', type=str, default='triplet',
                        help='save names of model and log')
    parser.add_argument('--num-epoches', dest='num_epoches', type=int, default=10,
                        help='the number of training epochs')

    parser.add_argument('--fine_tune',dest='fine_tune',default=False, action='store_true',
                        help='If fine-tune or not')
    parser.add_argument('--ignore_neg',dest='ignore_neg',default=False, action='store_true',
                        help='If ignore negative or not')
    parser.add_argument('--fine_tune_all',dest='fine_tune_all',default=False, action='store_true',
                        help='If fine-tune starting from conv4_ or not')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    model_dir = '../../models/SINT_pretrained/'

    args = parse_args()
    batch_size = args.batch_size
    num_epoch = args.num_epoches
    num_rois = args.num_rois
    devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    if args.fine_tune_all:
        if args.ignore_neg:
            prefix = os.path.join('./all_models', 'ignore_threshold_' + str(args.lmnn_threshd) + '_lr_' + str(args.lr))
        else:
            prefix = os.path.join('./all_models', 'threshold_' + str(args.lmnn_threshd)+'_lr_'+str(args.lr))
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        prefix = prefix+'/SINT'

        if args.ignore_neg:
            logdir = './all_logs/ignore_learning_rate_' + str(args.lr) + '_threshold_' + str(args.lmnn_threshd) + '_logs/'
        else:
            logdir = './all_logs/learning_rate_'+str(args.lr)+'_threshold_'+str(args.lmnn_threshd)+'_logs/'
    else:
        if args.ignore_neg:
            prefix = os.path.join('./models', 'ignore_threshold_' + str(args.lmnn_threshd) + '_lr_' + str(args.lr))
        else:
            prefix = os.path.join('./models', 'threshold_' + str(args.lmnn_threshd)+'_lr_'+str(args.lr))
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        prefix = prefix+'/SINT'

        if args.ignore_neg:
            logdir = './logs/ignore_learning_rate_' + str(args.lr) + '_threshold_' + str(args.lmnn_threshd) + '_logs/'
        else:
            logdir = './logs/learning_rate_'+str(args.lr)+'_threshold_'+str(args.lmnn_threshd)+'_logs/'

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        cmd = 'rm -rf '+logdir+'*'
        os.system(cmd) # clear log files

    # set logging file
    logging.basicConfig(filename=os.path.join(logdir,'{}.log'.format(args.mode)),level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info(args)

    # where to keep your TensorBoard logging file
    summary_writer = SummaryWriter(logdir)

    # set model
    _, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, 'SINT'), 0)

    fixed_param_names = []
    if args.fine_tune:
        for param in arg_params:
            if not 'fc' in param and not 'conv5_3' in param:
                fixed_param_names.append(param)

    if args.fine_tune_all:
        for param in arg_params:
            if 'conv1_' in param or 'conv2_' in param or 'conv3_' in param:
                fixed_param_names.append(param)

    fixed_param_names.sort()

    train, val, num_labels, num_samples = get_data_iter()

    # while(1):
    #     train.next()

    net = get_symbol(num_labels, args)
    mx.viz.plot_network(net).view()
    # net = build_network(symbol, num_labels)
    output_names = net.list_outputs()
    print output_names
    stepPerEpoch = int(num_samples / batch_size)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
        step=[stepPerEpoch * x for x in [3,6]], factor=0.5)
    init = mx.initializer.Xavier(
        rnd_type='gaussian', factor_type='in', magnitude=2)

    print 'num_labels: {}'.format(num_labels)
    print 'num_samples: {}'.format(num_samples)
    print 'step per epoch: {}'.format(stepPerEpoch)

    num = 1
    if args.lmnn:
        num += 1

    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    batch_end_callback = mx.callback.Speedometer(batch_size=batch_size)

    optimizer_params = (('learning_rate', args.lr), ('momentum', 0.9),
                        ('wd',0.0001), ('clip_gradient',10),('lr_scheduler',lr_scheduler),
                        ('rescale_grad',1.0 / batch_size))

    model = MutableModule(symbol=net,
                          data_names = train.data_names,
                          label_names=train.label_names,
                          logger=logger,
                          context=devices,
                          fixed_param_names=fixed_param_names)

    # model = mx.mod.Module(symbol=net,
    #                       data_names = train.data_names,
    #                       label_names=train.label_names,
    #                       logger=logger,
    #                       context=devices,
    #                       fixed_param_names=fixed_param_names)

    model.fit(train_data=train,
              eval_metric=None,
              eval_data=val,
              validation_metric=None,
              optimizer='sgd',
              optimizer_params= optimizer_params,
              allow_missing=True,
              initializer=init,
              arg_params=arg_params,
              aux_params=aux_params,
              num_epoch=num_epoch,
              epoch_end_callback = epoch_end_callback,
              batch_end_callback=batch_end_callback,
              summary_writer=summary_writer
              )
