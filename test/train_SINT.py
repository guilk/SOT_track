import os
import mxnet as mx
import numpy as np

if __name__ == '__main__':
    model_dir = '../models/SINT_pretrained/'
    mean_file = np.load('../models/SINT/ilsvrc_2012_mean.npy').mean(1).mean(1)

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, 'SINT'), 0)

    print sym.get_internals()

