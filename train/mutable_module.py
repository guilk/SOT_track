import logging

from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module
from mxnet.model import BatchEndParam
from mxnet import metric
from mxnet import ndarray
import time
import numpy as np

def _as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def check_label_shapes(labels, preds, shape=0):
    """Check to see if the two arrays are the same size."""

    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))



class IgnoreAccuracy(metric.EvalMetric):
    '''
    Compute accuracy without considering the ignored labels
    '''

    def __init__(self, axis=1, name='ignore_accuracy',
                 output_names=None, label_names=None):
        super(IgnoreAccuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            print 'pred: ',pred_label
            print 'gt: ',label
            check_label_shapes(label, pred_label)

            keep_inds = np.where(label != 0)
            pred_label = pred_label[keep_inds]
            label = label[keep_inds]

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

class MutableModule(Module):


    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, summary_writer = None):

        assert num_epoch is not None, 'please specify number of epochs'
        self.num_batch = 0
        self.writer = summary_writer

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)

        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        acc_metric = IgnoreAccuracy(output_names=['softmax_output'], label_names=['softmax_label'])
        # acc_metric = metric.Accuracy(output_names=['softmax_output'], label_names=['softmax_label'])
        lmnn_metric = metric.Loss(output_names=['lmnn_output'], label_names=['softmax_label'])

        if validation_metric is None:
            validation_metric = lmnn_metric

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            acc_metric.reset()
            lmnn_metric.reset()
            # eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)

                self.update()

                self.update_metric(acc_metric, data_batch.label)
                self.update_metric(lmnn_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

                # one epoch of training is finished
                for name, val in acc_metric.get_name_value():
                    self.logger.info('Epoch[%d] Accuracy Train-%s=%f', epoch, name, val)
                for name, val in lmnn_metric.get_name_value():
                    self.logger.info('Epoch[%d] Lmnn Train-%s=%f', epoch, name, val)

                if self.num_batch % 10 == 0:
                    # print acc_metric.sum_metric, acc_metric.num_inst
                    self.writer.add_scalar('{}/cls_acc'.format('Train'), acc_metric.sum_metric / acc_metric.num_inst,
                                                       self.num_batch)
                    self.writer.add_scalar('{}/lmnn_loss'.format('Train'), lmnn_metric.sum_metric / lmnn_metric.num_inst,
                                                       self.num_batch)

                self.num_batch += 1

            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()