import os
import numpy as np
import glob
import ntpath
import random
import cPickle
import cv2
import argparse
import itertools



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='prepare images and boxes for training')

    parser.add_argument('--split_type', dest='split_type',
                        help='the name of data split', default='val', type=str)
    parser.add_argument('--crop_image',dest='crop_image',action='store_true',default=False,
                        help='Crop image or not')
    parser.add_argument('--crop_size', dest='crop_size',
                        help='the size of cropped image', default=224, type=int)
    args = parser.parse_args()

    return args

def crop_image(pos_boxes, neg_boxes, crop_size):
    boxes = np.concatenate((pos_boxes, neg_boxes),axis=0)
    boxes_info = np.tile(np.zeros((1,4)),(boxes.shape[0],1))
    boxes = np.concatenate((boxes, boxes_info), axis = 1)

    im_h = boxes[0,-6]
    im_w = boxes[0,-5]
    img_index = list(set(boxes[:,5].tolist()))
    assert len(img_index) == 2

    for index in img_index:
        cand = boxes[:,5] == index

        min_x = np.min(boxes[cand, 0])
        max_x = np.max(boxes[cand, 2])
        min_y = np.min(boxes[cand, 1])
        max_y = np.max(boxes[cand, 3])

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        new_size = max(bbox_h, bbox_w)
        # if bbox_w > crop_size or bbox_h > crop_size:
        #     new_size = max(bbox_h, bbox_w)
        # else:
        #     new_size = crop_size

        if min_x + new_size > im_w:
            min_x = im_w - new_size
        if min_y + new_size > im_h:
            min_y = im_h - new_size

        crop_x1 = max(min_x,0)
        crop_y1 = max(min_y,0)
        crop_x2 = min(crop_x1 + new_size, im_w)
        crop_y2 = min(crop_y1 + new_size, im_h)

        boxes[cand,-4] = crop_x1
        boxes[cand,-3] = crop_y1
        boxes[cand,-2] = crop_x2
        boxes[cand,-1] = crop_y2
    half_size = boxes.shape[0] / 2
    pos_boxes = boxes[:half_size,:].copy()
    neg_boxes = boxes[half_size:,:].copy()

    return pos_boxes,neg_boxes

if __name__ == '__main__':
    args = parse_args()
    split_type = args.split_type
    # crop_image = args.crop_image
    crop_size = args.crop_size

    src_root = '/data01/ALOV/{}_metafile_for_gen_hdf5_trial01'.format(split_type)
    dst_root = '/data01/ALOV/triplet/triplet_{}'.format(split_type)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    db_root = '/data01/ALOV/'
    img_root = '/data01/ALOV/images/'
    ann_root = '/data01/ALOV/annotations/'
    if split_type == 'train':
        num_pairs = 71552
    else:
        num_pairs = 7684

    rseed = 10
    threshold = 0.5
    random.seed(rseed)

    image_info_1 = []
    image_info_2 = []

    nr_box_perim = 64
    # box_info = np.zeros((num_pairs, nr_box_perim * 5 * 2 + 2))
    box_info = []
    image_pair_txt_files = glob.glob(os.path.join(src_root, '*_image_pairs.txt'))
    image_pair_txt_files.sort()

    seq_dict = {}
    seq_set = set()
    stat_dict = {}

    count = 0
    for ind, image_pair_txt_file in enumerate(image_pair_txt_files):
        fr = open(image_pair_txt_file, 'rb')
        lines = fr.readlines()
        if len(lines) == 0:
            continue

        pair_file_name = ntpath.basename(image_pair_txt_file)
        splits = pair_file_name.split('_')[:2]

        video_name = '_'.join(splits)
        if video_name not in seq_dict:
            seq_dict[video_name] = ind+1
            stat_dict[video_name] = len(lines)
            seq_set.add(video_name)
        else:
            stat_dict[video_name] = stat_dict[video_name] + len(lines)

        for line in lines:
            line_splits = line.rstrip('\r\n').split(' ')
            image_info_1.append(video_name+'_'+line_splits[0])
            image_info_2.append(video_name+'_'+line_splits[1])

        box_file_name = '_'.join(splits)+'_box_pairs.txt'
        box_file_path = os.path.join(src_root, box_file_name)
        # print box_file_path
        boxes = np.loadtxt(box_file_path, delimiter=' ')

        if boxes.ndim == 1:
            boxes = boxes[np.newaxis,:]
        assert boxes.shape[0] == len(lines)
        assert boxes.shape[1] == 642
        # box_info[count:count+len(lines),:] = boxes.copy()
        box_info.append(boxes.copy())
        count += len(lines)

    box_info = np.concatenate(box_info)

    imgs_info = image_info_1 + image_info_2
    imgs_info = list(set(imgs_info))
    imgs_info.sort()
    imgs_ind = range(len(imgs_info))
    imgs_dict = dict(zip(imgs_info, imgs_ind))

    assert count == num_pairs
    index_shuf = range(num_pairs)

    order = 32
    max_num = 8
    pos_order = order / 4
    neg_order = order / 2
    valid_threshold = 0.1

    pos_data = []
    neg_data = []

    seq_target = set()
    stat_dict = {}
    for i,idx in enumerate(index_shuf):

        image_info_1_part = image_info_1[idx]
        path_splits = image_info_1_part.split('_')
        img1_seq_name = path_splits[0] + '_' + path_splits[1]
        img_path = os.path.join(img_root, path_splits[0],
                                path_splits[0] + '_' + path_splits[1], path_splits[2])

        im = cv2.imread(img_path)
        im1_h = im.shape[0]
        im1_w = im.shape[1]

        # print image_info_1_part
        image_info_2_part = image_info_2[idx]
        path_splits = image_info_2_part.split('_')
        img2_seq_name = path_splits[0] + '_' + path_splits[1]
        img_path = os.path.join(img_root, path_splits[0],
                                path_splits[0] + '_' + path_splits[1], path_splits[2])

        print img_path

        im = cv2.imread(img_path)
        im2_h = im.shape[0]
        im2_w = im.shape[1]

        box_info_part = box_info[idx,:].copy()

        num_box1 = int(box_info_part[0])
        num_box2 = int(box_info_part[1])

        binfo = box_info_part[2:]
        # half_size = binfo.shape[0] / 2
        binfo1 = binfo[:num_box1*5].copy()
        binfo2 = binfo[num_box1*5:(num_box1+num_box2)*5].copy()
        boxes1 = np.reshape(binfo1, (-1,5))
        boxes2 = np.reshape(binfo2, (-1,5))

        # filter
        # boxes1 = boxes1[boxes1[:,4] > valid_threshold,:].copy()
        # boxes2 = boxes2[boxes2[:,4] > valid_threshold,:].copy()

        # boxes1 = boxes1[boxes1[:,4] < valid_threshold,:].copy()
        # boxes2 = boxes2[boxes2[:,4] < valid_threshold,:].copy()

        boxes1_info = np.tile(
            np.asarray([imgs_dict[image_info_1_part],i,seq_dict[img1_seq_name],im1_h,im1_w]).reshape((1,5)),(boxes1.shape[0],1))
        boxes1 = np.concatenate((boxes1, boxes1_info),axis=1)

        boxes2_info = np.tile(
            np.asarray([imgs_dict[image_info_2_part],i,seq_dict[img2_seq_name],im2_h,im2_w]).reshape((1,5)),(boxes2.shape[0],1))
        boxes2 = np.concatenate((boxes2, boxes2_info), axis=1)

        pos_boxes1 = boxes1[boxes1[:,4] >= threshold,:].copy()
        pos_boxes2 = boxes2[boxes2[:,4] >= threshold,:].copy()
        neg_boxes1 = boxes1[boxes1[:,4] < threshold, :].copy()
        neg_boxes2 = boxes2[boxes2[:,4] < threshold, :].copy()
        neg_boxes = np.concatenate((neg_boxes1, neg_boxes2), axis=0)
        neg_boxes = neg_boxes[neg_boxes[:,4] < valid_threshold,:].copy()


        pos_boxes1_ind = range(pos_boxes1.shape[0])
        pos_boxes2_ind = range(pos_boxes2.shape[0])

        comb_pos = list(itertools.product(pos_boxes1_ind, pos_boxes2_ind))
        num_pos = min(len(comb_pos),max_num) / pos_order * pos_order
        if num_pos == 0:
            continue

        num_neg = num_pos * 2
        # print num_neg , neg_boxes.shape[0]
        if num_neg > neg_boxes.shape[0]:
            neg_boxes = np.concatenate((neg_boxes.copy(),neg_boxes.copy()),axis=0)
            if num_neg > neg_boxes.shape[0]:
                print num_neg,neg_boxes.shape[0]
                continue

        random.shuffle(comb_pos)
        np.random.shuffle(neg_boxes)
        pos_boxes = np.zeros((num_pos*2,boxes1.shape[1]))
        for pos_ind in range(num_pos):
            pos_boxes[2*pos_ind,:] = pos_boxes1[comb_pos[pos_ind][0],:]
            pos_boxes[2*pos_ind+1,:] = pos_boxes2[comb_pos[pos_ind][1],:]
        neg_boxes = neg_boxes[:num_neg,:].copy()
        pos_boxes,neg_boxes = crop_image(pos_boxes,neg_boxes,crop_size)
        pos_data.append(pos_boxes)
        neg_data.append(neg_boxes)

        if img2_seq_name not in stat_dict:
            stat_dict[img2_seq_name] = pos_boxes.shape[0] + neg_boxes.shape[0]
        else:
            stat_dict[img2_seq_name] += pos_boxes.shape[0] + neg_boxes.shape[0]

    pos_data = np.concatenate(pos_data)
    neg_data = np.concatenate(neg_data)
    assert pos_data.shape[0] == neg_data.shape[0]
    print stat_dict
    print len(stat_dict)
    print pos_data.shape[0]
    print neg_data.shape[0]

    save_img_file = os.path.join(dst_root, 'img_list.pkl')
    with open(save_img_file, 'wb') as fid:
        cPickle.dump(imgs_info, fid, cPickle.HIGHEST_PROTOCOL)

    save_pos_file = os.path.join(dst_root, 'pos_data.pkl')
    with open(save_pos_file, 'wb') as fid:
        cPickle.dump(pos_data, fid, cPickle.HIGHEST_PROTOCOL)

    save_neg_file = os.path.join(dst_root, 'neg_data.pkl')
    with open(save_neg_file, 'wb') as fid:
        cPickle.dump(neg_data, fid, cPickle.HIGHEST_PROTOCOL)
