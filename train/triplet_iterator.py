import mxnet as mx
import numpy as np
import os
import cPickle
import cv2
import random
from mxnet.io import DataIter, DataBatch

class TripletIter(mx.io.DataIter):
    '''
    Data iterator
    Used for triplet loss and lmnn loss
    '''
    def __init__(self, split_type, aug_params):
        super(TripletIter,self).__init__()
        self.db_root = aug_params.db_root
        self.imageSz = aug_params.imageSz
        self.db_name = aug_params.db_name
        self.num_rois = aug_params.num_rois
        self.sampling = aug_params.sampling
        self.rand_mirror = aug_params.rand_mirror
        self.ignore_neg = aug_params.ignore_neg
        self.num_gpus = len([int(i) for i in aug_params.gpus.split(',')])

        self.triplet = aug_params.triplet
        self.lmnn = aug_params.lmnn

        self.counter = 0
        self.data_thres = 1600

        self.batch_size = aug_params.batch_size
        self.pair_batch_size = self.batch_size / 2
        self.shuffle = aug_params.shuffle
        self.img_root = os.path.join(self.db_root, 'images')
        self.mean = np.array((103.939, 116.779, 123.68)).reshape((1, 1, 3))  # VGG-16 mean BGR
        self.split_type = split_type

        self.img_list, self.full_pos_data, self.full_neg_data = self._get_img_list()
        self.reset()
        self.num_data = self.pos_data.shape[0]
        self.img_db = self.get_img_db()

        self.data_names = ['data', 'rois']
        self.label_names = ['softmax_label']


    def _get_img_list(self):
        '''
        data: 
            bbox: x1,y1,x2,t2, IoU, 
                image_index(index used for retriving image from dict), 
                pair_index(used when shuffling data), 
                seq_index(used as labels), im_h, im_w,
                cropped_x1, cropped_y1, cropped_x2, cropped_y2
        :return: 
        '''

        bbox_pos_file = os.path.join(self.db_root, 'triplet', self.split_type, 'pos_data.pkl')
        with open(bbox_pos_file, 'rb') as fid:
            pos_data = cPickle.load(fid)

        bbox_neg_file = os.path.join(self.db_root, 'triplet', self.split_type, 'neg_data.pkl')
        with open(bbox_neg_file, 'rb') as fid:
            neg_data = cPickle.load(fid)

        img_info_file = os.path.join(self.db_root, 'triplet', self.split_type, 'img_list.pkl')
        with open(img_info_file, 'rb') as fid:
            img_list = cPickle.load(fid)

        return img_list, pos_data, neg_data

    def get_img_db(self):
        '''
        Load image databaset to memory and save to a dict
        :return: imdb[img_path] = H x W x 3
        '''
        cache_file = os.path.join(self.db_root, 'triplet', self.split_type, 'imgs_db.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                img_db = cPickle.load(fid)
            print '{} images db loaded from {}'.format(self.db_name, cache_file)
            return img_db

        img_db = {}
        for img_name in self.img_list:
            path_splits = img_name.split('_')
            img_path = os.path.join(self.img_root, path_splits[0],
                                    path_splits[0]+'_'+path_splits[1], path_splits[2])
            print img_path
            img = cv2.imread(img_path)
            img_db[img_name] = img

        with open(cache_file, 'wb') as fid:
            cPickle.dump(img_db, fid, cPickle.HIGHEST_PROTOCOL)

        print 'write images db to {}'.format(cache_file)
        return img_db

    def _process_data(self):

        max_label_value = max(set(self.pos_data[:,7].tolist()))
        if self.ignore_neg:
            self.neg_data[self.neg_data[:, 4] < 0.5, 7] = 0
        else:
            self.neg_data[self.neg_data[:,4] > 0.3, 7] = 0
            self.neg_data[self.neg_data[:, 4] < 0.3, 7] = max_label_value + self.neg_data[self.neg_data[:, 4] < 0.3, 7]

        self.num_labels = len(set(self.pos_data[:,7].tolist() + self.neg_data[:,7].tolist()))
        self.num_data = self.pos_data.shape[0]


    def get_stats(self):
        return self.num_labels, self.pos_data.shape[0] * 2

    @property
    def provide_data(self):
        return [('data', (self.batch_size, 3, self.imageSz, self.imageSz)),
                ('rois', (self.batch_size, 1, 5))]

    @property
    def provide_label(self):
        labels = [('softmax_label', (self.batch_size,))]
        return labels

    def _process(self, img, bbox_info):
        '''
        :param img: RGB image of H x W x 3 
        :return: RGB image of H x W x 3, after resize and mean subtraction
        '''

        # make sure the dimension of input image is 3D
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Crop image based on the crop bounding box
        crop_x1 = bbox_info[-4]
        crop_y1 = bbox_info[-3]
        crop_x2 = bbox_info[-2]
        crop_y2 = bbox_info[-1]
        img = img[int(crop_y1):int(crop_y2),int(crop_x1):int(crop_x2)].copy()

        im_h = img.shape[0]
        im_w = img.shape[1]

        img = np.array(img, dtype=np.float32)
        img = cv2.resize(img, (self.imageSz, self.imageSz),
                         interpolation=cv2.INTER_NEAREST)
        img = img - self.mean
        img[:,:,:] = img[:,:,[2,1,0]]
        img = img.transpose((2,0,1))

        # resized bbox
        bbox = np.zeros((4,))
        bbox[0] = (bbox_info[0] - crop_x1) * self.imageSz / im_w
        bbox[1] = (bbox_info[1] - crop_y1) * self.imageSz / im_h
        bbox[2] = (bbox_info[2] - crop_x1) * self.imageSz / im_w
        bbox[3] = (bbox_info[3] - crop_y1) * self.imageSz / im_h

        # Rand mirror images
        rand_flip = random.randint(0,1)
        if self.rand_mirror and rand_flip == 1:
            img = img[:,:,::-1]
            bbox[0] = self.imageSz - bbox[2] - 1
            bbox[2] = self.imageSz - bbox[0] - 1

        return img,bbox



    def reset(self):
        self.cursor = 0

        # For positive data, use pair-wise data shuffle based on pair index
        labels = list(set(self.full_pos_data[:, 7].tolist()))
        img_pair_index = self.full_pos_data[:,6].tolist()
        s = [(x, int(random.random() * 1e7), i) for i,x in enumerate(img_pair_index) if i % 2 == 0]
        s = sorted(s, key=lambda x:(x[0],x[1]))
        lst = [x[2] for x in s]
        pos_lst = []
        for ind in lst:
            pos_lst.append(ind)
            pos_lst.append(ind+1)
        self.full_pos_data = self.full_pos_data[pos_lst,:].copy()

        img_pair_index = self.full_neg_data[:,6].tolist()
        s = [(x, int(random.random() * 1e7), i) for i, x in enumerate(img_pair_index)]
        s = sorted(s, key=lambda x:(x[0],x[1]))
        neg_lst = [x[2] for x in s]
        self.full_neg_data = self.full_neg_data[neg_lst,:].copy()


        # Downsample data to achieve data balance
        pos_data_lst = []
        neg_data_lst = []
        for label in labels:
            pos_data = self.full_pos_data[self.full_pos_data[:,7] == label,:].copy()
            pos_data_lst.append(pos_data[:self.data_thres,:].copy())

            neg_data = self.full_neg_data[self.full_neg_data[:,7] == label,:].copy()
            neg_data_lst.append(neg_data[:self.data_thres,:].copy())

        self.pos_data = np.concatenate(pos_data_lst)
        self.neg_data = np.concatenate(neg_data_lst)


        # Bacth-wise data shuffle
        idx = range(0, self.pos_data.shape[0], self.pair_batch_size)
        random.shuffle(idx)
        ret = []
        for i_ind in idx:
            for batch_ind in range(self.pair_batch_size):
                ret.append(i_ind + batch_ind)

        self.pos_data = self.pos_data[ret,:].copy()
        self.neg_data = self.neg_data[ret,:].copy()
        assert (self.pos_data[:, 7] == self.neg_data[:, 7]).all()
        self._process_data()


    def _fetch_data(self):
        '''
        The first half of batches: positive image patch pairs (IoUs > 0.5)
        The second half of batches: negative image patches (IoUs < 0.5, sampled from the same image pairs) 
        :return: [data,bbox],label
        '''

        data = np.zeros((self.batch_size, 3, self.imageSz, self.imageSz))
        bbox = np.zeros((self.batch_size, 1, 5))
        softmax_label = np.ones((self.batch_size,))

        for idx in range(self.pair_batch_size):
            index = self.cursor + idx
            data[idx],input_bbox = self._process(self.img_db[self.img_list[int(self.pos_data[index,5])]], self.pos_data[index])
            bbox[idx,0,1:] = input_bbox.copy()
            bbox[idx, 0, 0] = idx

        for idx in range(self.pair_batch_size):

            index = self.cursor + idx
            data[idx + self.pair_batch_size],input_bbox = self._process(self.img_db[self.img_list[int(self.neg_data[index,5])]], self.neg_data[index])
            bbox[idx+self.pair_batch_size,0,1:] = input_bbox.copy()
            bbox[idx + self.pair_batch_size, 0, 0] = idx + self.pair_batch_size


        softmax_label[:self.pair_batch_size] = self.pos_data[self.cursor : self.cursor + self.pair_batch_size,7].copy().astype(np.int16)
        softmax_label[self.pair_batch_size:] = self.neg_data[self.cursor: self.cursor + self.pair_batch_size, 7].copy().astype(np.int16)

        return [mx.nd.array(data), mx.nd.array(bbox)], [mx.nd.array(softmax_label)]

    def next(self):
        if self.cursor + self.pair_batch_size >= self.num_data:
            raise StopIteration

        data,label = self._fetch_data()
        self.cursor += self.pair_batch_size

        return DataBatch(data=data, label=label,
                         pad=0, index=None,
                         provide_data=self.provide_data,
                         provide_label=self.provide_label)


