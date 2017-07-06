import os
import numpy as np
import datetime
import image_io
import random
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.linear_model import Ridge
    import scipy.io as sio
import argparse
import mxnet as mx
import cv2
import cPickle as pickle
from LRU import *

from collections import namedtuple
Batch = namedtuple('Batch',['data'])

def preproc_img(img, mean, imageSz):
    img = img.astype(np.float32, copy=False)
    img = image_io.resize_image(img, (imageSz, imageSz))
    img = img.transpose((2,0,1))
    img = img[(2,1,0), :, :]
    img *= 255 # input image is in range(0,1)
    if mean.ndim == 1:
        mean = mean[:,np.newaxis, np.newaxis]
    img -= mean
    img = img[(2,1,0), :, :]
    img = img[np.newaxis, :]
    return img

def func_iou(bb, gtbb):
    iou = 0
    iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1
    ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1

    if iw>0 and ih>0:
        ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih
        iou = iw*ih/ua

    return iou


# samples_templates = sample_regions_precompute(float(base_rad) / 512 * firstframe.shape[1], nr_angles, 1,
                                              # scales=[0.7, 0.8, 0.9, 1, 1 / 0.9, 1 / 0.8, 1 / 0.7])
def sample_regions_precompute(rad, nr_ang, stepsize, scales=[0.7071, 1, 1.4142]):

    nr_step = int(rad / stepsize)

    cos_values = np.cos(np.arange(0,2*np.pi,2*np.pi/nr_ang))
    sin_values = np.sin(np.arange(0,2*np.pi,2*np.pi/nr_ang))

    dxdys = np.zeros((2,nr_step*nr_ang+1))
    count = 0
    for ir in range(1,nr_step+1):
        offset = stepsize * ir
        for ia in range(1,nr_ang+1):

            dx = offset * cos_values[ia-1]
            dy = offset * sin_values[ia-1]
            count += 1
            dxdys[0, count-1] = dx
            dxdys[1, count-1] = dy

    samples = np.zeros((4,(nr_ang*nr_step+1)*len(scales)))
    count = 0
    jump = nr_step * nr_ang+1
    for s in scales:
        samples[0:2, count*jump:(count+1)*jump] = dxdys
        samples[2, count*jump:(count+1)*jump] = s
        samples[3, count*jump:(count+1)*jump] = s
        count = count + 1

    return samples # dx dy 1*s 1*s


# samples = sample_regions(init_box[0], init_box[1], init_box[2], init_box[3], firstframe.shape[1],
                         # firstframe.shape[0], samples_templates)
# samples_templates: dx dy 1*s 1*s

def sample_regions(x, y, w, h, im_w, im_h, samples_template):

    samples = samples_template.copy()
    samples[0,:] += x
    samples[1,:] += y
    samples[2,:] *= w
    samples[3,:] *= h

    samples[2,:] = samples[0,:] + samples[2,:] - 1
    samples[3,:] = samples[1,:] + samples[3,:] - 1
    samples = np.round(samples)


    flags = np.logical_and(np.logical_and(np.logical_and(samples[0,:]>0, samples[1,:]>0), samples[2,:]<im_w), samples[3,:]<im_h)
    samples = samples[:,flags]

    return samples # x1 y1 x2 y2



def parse_args():

        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='eval_OTB')

        parser.add_argument('--overlapthresh', dest='ov_thresh', help='overlap threshold to select training samples for box regression', default=0.6, type=float)
        parser.add_argument('--numangles', dest='nr_angles', help='num of angles for angular sampling for box regression', default=20, type=int)
        parser.add_argument('--topk', dest='topk', help='num of top samples for flow checking', default=5, type=int)
        parser.add_argument('--flowtheta', dest='theta', help='threshold for flow inconsistency checking', default=0.25, type=float)

        args = parser.parse_args()

        return args

if __name__ == '__main__':

    args = parse_args()
    nr_angles = args.nr_angles
    ov_thresh = args.ov_thresh
    topk = args.topk
    theta = args.theta

    srcRoot = '../OTB/'  # dataset root
    optical_flow_root = '/data01/SOT/OTB_epicflow/'
    # saveRoot = '../results/mxnet_opticalflow_tracking_results/'
    saveRoot = '../results/mxnet_epicflow_cache_ratio_0.75_threshold_0.9/'
    if not os.path.exists(saveRoot):
        os.makedirs(saveRoot)
    img_save_root = '/data01/SOT/imgs'
    # saveRoot = '../results/test/'
    input_img_root = './mxnet_imgs'

    imageSz = 512
    D = 25088 * 2 + 4096
    base_rad = 30
    stepsize = 3
    use_flow = 0.6
    cache_size = 5
    gpu_id = 2

    signature = '_OTB_eval'

    model_dir = '../models/converted_SINT/'
    mean_file = np.load('../models/SINT/ilsvrc_2012_mean.npy').mean(1).mean(1)
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir,'SINT'),0)

    videos = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', 'Coke', 'Couple', 'Crossing', 'David2', 'David3', 'David', 'Deer', 'Dog1', 'Doll', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping', 'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking2', 'Walking', 'Woman', 'Jogging']
    # videos = ['Car4']
    num_videos = len(videos)
    seen_jogging = False
    model = mx.mod.Module(symbol=sym.get_internals()['feat_l2_output'], data_names=['data', 'rois'], label_names=[],
                          context=mx.gpu(gpu_id))

    for vid in range(num_videos):

        base_sim = 0
        base_counter = 0
        base_update = True
        videoname = videos[vid]
        savefile = saveRoot + videoname + signature + '.txt'
        templ_list = []

        dst_folder = os.path.join(img_save_root, videoname)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        ##################### first frame
        gtfile = srcRoot + videoname + '/' + 'groundtruth_rect.txt'
        if os.path.exists(gtfile) == False:
            if seen_jogging == False:
                gtfile = srcRoot + videoname + '/' + 'groundtruth_rect.1.txt'
                savefile = saveRoot + videoname + '-1' + signature + '.txt'
                seen_jogging = True
            else:
                gtfile = srcRoot + videoname + '/' + 'groundtruth_rect.2.txt'
                savefile = saveRoot + videoname + '-2' + signature + '.txt'

        if os.path.exists(savefile) == True:
            continue

        try:
            gtboxes = np.loadtxt(gtfile, delimiter=',')  # x y w h
        except:
            gtboxes = np.loadtxt(gtfile)

        if videoname == 'Tiger1':  # in OTB, for 'Tiger1', the starting frame is the 6th
            gtboxes = gtboxes[5:, :]

        firstframename = '0001.jpg'
        if videoname == 'David':
            firstframename = '0300.jpg'
        if videoname == 'Tiger1':  ##################################
            firstframename = '0006.jpg'

        img_path = srcRoot + videoname + '/' + 'img/' + firstframename

        firstframe = image_io.load_image(img_path)

        init_box = gtboxes[0, :].copy()  # x y w h
        nr_frames2track = gtboxes.shape[0]  # including first frame

        input_roi = np.zeros((1, 5))
        input_roi[0, 1] = init_box[0] * imageSz / firstframe.shape[1]
        input_roi[0, 2] = init_box[1] * imageSz / firstframe.shape[0]
        input_roi[0, 3] = (init_box[0] + init_box[2] - 1) * imageSz / firstframe.shape[1]
        input_roi[0, 4] = (init_box[1] + init_box[3] - 1) * imageSz / firstframe.shape[0]
        input_roi[0, 1:] -= 1  # starting from 0
        input_roi = input_roi[np.newaxis,:]

        model.bind(for_training=False, data_shapes=[('data', (1, 3, imageSz, imageSz)), ('rois', (1, input_roi.shape[1], 5))], force_rebind=True)
        model.set_params(arg_params, aux_params)
        img = preproc_img(firstframe, mean_file, imageSz)
        model.forward(Batch([mx.nd.array(img), mx.nd.array(input_roi)]))
        qfeat = model.get_outputs()[0].T
        templ_LRU = LRUCache(qfeat, gpu_id)

        ########################################box regression (training)#####################################
        samples_templates = sample_regions_precompute(float(base_rad) / 512 * firstframe.shape[1], nr_angles, 1,
                                                      scales=[0.7, 0.8, 0.9, 1, 1 / 0.9, 1 / 0.8, 1 / 0.7])
        samples = sample_regions(init_box[0], init_box[1], init_box[2], init_box[3], firstframe.shape[1],
                                 firstframe.shape[0], samples_templates)
        # pdb.set_trace()
        ov_samples = np.zeros((1, samples.shape[1]))
        init_box_ = init_box.copy()
        init_box_[2] = init_box_[2] + init_box_[0] - 1
        init_box_[3] = init_box_[3] + init_box_[1] - 1

        #########################################add template
        img_draw = cv2.imread(img_path, -1)
        img_roi = img_draw[int(init_box_[1]):int(init_box_[3]),
                  int(init_box_[0]):int(init_box_[2])]
        templ_list.append(img_roi)

        for ii in range(0, samples.shape[1]):
            ov_samples[0, ii] = func_iou(samples[:, ii], init_box_)

        sel_samples = samples[:, ov_samples[0, :] > ov_thresh]
        sel_rois = np.zeros((sel_samples.shape[1], 5))
        sel_rois[:, 1:] = np.transpose(sel_samples).copy()
        sel_rois[:, 1] = sel_rois[:, 1] * imageSz / firstframe.shape[1] - 1
        sel_rois[:, 3] = sel_rois[:, 3] * imageSz / firstframe.shape[1] - 1
        sel_rois[:, 2] = sel_rois[:, 2] * imageSz / firstframe.shape[0] - 1
        sel_rois[:, 4] = sel_rois[:, 4] * imageSz / firstframe.shape[0] - 1
        sel_rois = sel_rois[np.newaxis,:]

        model.bind(for_training=False, data_shapes=[('data', (1, 3, imageSz, imageSz)), ('rois', (1, sel_rois.shape[1], 5))], force_rebind=True)
        model.set_params(arg_params, aux_params)
        model.forward(Batch([mx.nd.array(img), mx.nd.array(sel_rois)]))
        br_feats = model.get_outputs()[0].asnumpy().squeeze()
        br_feats = br_feats[:,0:25088]
        br_coor = sel_samples.copy()
        br_coor[2, :] = br_coor[2, :] - br_coor[0, :] + 1  # w
        br_coor[3, :] = br_coor[3, :] - br_coor[1, :] + 1  # h
        br_coor[0, :] = br_coor[0, :] + 0.5 * br_coor[2, :]
        br_coor[1, :] = br_coor[1, :] + 0.5 * br_coor[3, :]

        gt_coor = init_box.copy()
        gt_coor[0] = gt_coor[0] + 0.5 * gt_coor[2]
        gt_coor[1] = gt_coor[1] + 0.5 * gt_coor[3]

        target_x = np.divide((gt_coor[0] - br_coor[0, :]), br_coor[2, :])
        target_y = np.divide((gt_coor[1] - br_coor[1, :]), br_coor[3, :])
        target_w = np.log(np.divide(gt_coor[2], br_coor[2, :]))
        target_h = np.log(np.divide(gt_coor[3], br_coor[3, :]))

        ### learn regressor
        regr_x = Ridge(alpha=1, fit_intercept=False)
        regr_y = Ridge(alpha=1, fit_intercept=False)
        regr_w = Ridge(alpha=1, fit_intercept=False)
        regr_h = Ridge(alpha=1, fit_intercept=False)

        regr_x.fit(br_feats, target_x)
        regr_y.fit(br_feats, target_y)
        regr_w.fit(br_feats, target_w)
        regr_h.fit(br_feats, target_h)

        ################################################tracking
        frameidx = range(2, nr_frames2track + 1)
        if videoname == 'David':
            frameidx = range(301, nr_frames2track + 300)
        if videoname == 'Tiger1':
            frameidx = range(7, nr_frames2track + 6)

        prev_box = init_box  # x y w h

        maxboxes = np.zeros((nr_frames2track, 5))
        maxboxes[0, 0] = 3
        maxboxes[0, 1] = init_box[0]
        maxboxes[0, 2] = init_box[1]
        maxboxes[0, 3] = init_box[0] + init_box[2] - 1
        maxboxes[0, 4] = init_box[1] + init_box[3] - 1

        samples_tmpl = sample_regions_precompute(float(base_rad) / 512 * firstframe.shape[1],
                                                 10, stepsize)
        tmpl_size = samples_tmpl.shape[1]
        model.bind(for_training=False,
                   data_shapes=[('data', (1, 3, imageSz, imageSz)), ('rois', (1, tmpl_size, 5))], force_rebind=True)
        model.set_params(arg_params, aux_params)

        counter = 0
        starttime = datetime.datetime.now()
        for id in frameidx:
            framepath = '%s%s/img/%04d.jpg' % (srcRoot, videoname, id)
            dstFramePath = os.path.join(dst_folder, '%04d.jpg'%id)

            print framepath
            im = image_io.load_image(framepath)
            im_h = im.shape[0]
            im_w = im.shape[1]
            # x1 y1 x2 y2
            samples = sample_regions(prev_box[0], prev_box[1], init_box[2], init_box[3], im.shape[1], im.shape[0],
                                     samples_tmpl)
            nr_samples = samples.shape[1]
            rois = np.zeros((nr_samples, 5))
            rois[:, 1:] = np.transpose(samples).copy()
            rois[:, 1] = rois[:, 1] * imageSz / im_w - 1
            rois[:, 3] = rois[:, 3] * imageSz / im_w - 1
            rois[:, 2] = rois[:, 2] * imageSz / im_h - 1
            rois[:, 4] = rois[:, 4] * imageSz / im_h - 1
            if nr_samples < tmpl_size:
                invalid_rois = np.zeros((tmpl_size-nr_samples, 5))
                invalid_rois[:,:] = rois[-1,:]
                rois = np.concatenate((rois, invalid_rois), axis=0)
            rois = rois[np.newaxis,:]

            img = preproc_img(im, mean_file, imageSz)
            model.forward(Batch([mx.nd.array(img), mx.nd.array(rois)]))
            tfeats = model.get_outputs()[0]

            scores = mx.nd.dot(tfeats, qfeat).asnumpy().squeeze()[:nr_samples]

            max_idx = np.argmax(scores)

            # cur_sim = np.max(scores)
            if base_update:
                if base_counter == 1:
                    base_update = False
                else:
                    base_sim += scores[max_idx]
                    base_counter += 1

            # max_idx = np.argmax(scores)
            img_draw = cv2.imread(framepath, -1)
            # for ii in range(nr_samples):
            #     bb = samples[:, ii].copy()
            #     cv2.rectangle(img_draw, (int(bb[0]), int(bb[1])),
            #               (int(bb[2]), int(bb[3])), (255, 0, 0), 1)
            # cv2.imwrite(dstFramePath, img_draw)
            sim_ratio = scores[max_idx]/float(base_sim)

            if sim_ratio >= use_flow:
                # ################################################### flow
                if sim_ratio <= 0.75:
                    max_idx, scores, old_idx, isUpdate = templ_LRU.cal_sim(tfeats, nr_samples, scores, True)
                else:
                    max_idx, scores, old_idx, isUpdate = templ_LRU.cal_sim(tfeats, nr_samples, scores, False)

                # print samples[:,new_idx]
                # assert False

                # print img_draw.shape
                img_roi = img_draw[int(samples[1, max_idx]):int(samples[3, max_idx]), int(samples[0, max_idx]):int(samples[2, max_idx])]
                # print img_roi.shape
                if isUpdate == 2:
                    templ_list.append(img_roi.copy())
                elif isUpdate == 1:
                    templ_list[old_idx] = img_roi.copy()

                rank = np.argsort(scores)
                k = topk
                candidates = rank[-k:]
                ##optical flow
                px1 = int(prev_box[0])
                py1 = int(prev_box[1])
                px2 = int(prev_box[2] + prev_box[0] - 1)
                py2 = int(prev_box[3] + prev_box[1] - 1)

                # print prev_box
                flowfile = '%s%s/%04d.mat' % (optical_flow_root, videoname, id)
                try:
                    flow = sio.loadmat(flowfile)
                    flowvalue = flow['flow'].copy()
                    flow_x = flowvalue[:,:,0]
                    flow_y = flowvalue[:,:,1]
                    flow_x_region = flow_x[py1-1:py2, px1-1:px2].flatten() # row first
                    flow_y_region = flow_y[py1-1:py2, px1-1:px2].flatten()

                    xx,yy = np.meshgrid(np.arange(px1,px2+1,1),np.arange(py1,py2+1,1))
                    xx_flat = xx.flatten()
                    yy_flat = yy.flatten()

                    xx_next = xx_flat + flow_x_region
                    yy_next = yy_flat + flow_y_region

                    # pdb.set_trace()
                    pixel_count = np.zeros((k,))
                    for ii in range(k):
                        bb = samples[:,candidates[ii]].copy() # x1 y1 x2 y2
                        flags = xx_next >= bb[0]
                        flags = np.logical_and(flags, xx_next <= bb[2])
                        flags = np.logical_and(flags, yy_next >= bb[1])
                        flags = np.logical_and(flags, yy_next <= bb[3])
                        pixel_count[ii] = np.sum(flags)
                        cv2.rectangle(img_draw, (int(bb[0]), int(bb[1])),
                                      (int(bb[2]), int(bb[3])),(255, 0, 0), 2)
                    # theta = 0.15
                    passed = pixel_count > (prev_box[2]*prev_box[3]*theta)
                    if np.sum(passed) == 0:
                        max_idx = np.argmax(scores)
                    else:
                        candidates_left = candidates[passed]
                        for ii in range(len(candidates_left)):
                            bb = samples[:, candidates_left[ii]].copy()
                            cv2.rectangle(img_draw, (int(bb[0]), int(bb[1])),
                                          (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

                        max_idx = candidates_left[ np.argmax(scores[candidates_left])]
                except:
                    print 'could not read flow file: %s' % flowfile
                    max_idx = np.argmax(scores)
            else:
                print 'Not use optical flow: ' + str(scores[max_idx]/float(base_sim))
            cv2.imwrite(dstFramePath, img_draw)
            dstTemplPath = os.path.join(dst_folder, '%04d_template.pkl' % id)
            with open(dstTemplPath, 'wb') as outfile:
                pickle.dump(templ_list, outfile, pickle.HIGHEST_PROTOCOL)
            # np.savez(dstTemplPath, *templ_list)
            ###################################################

            prev_box = samples[:, max_idx].copy()
            maxboxes[counter + 1, 0] = scores[max_idx]

            prev_box[2] = prev_box[2] - prev_box[0] + 1  # w
            prev_box[3] = prev_box[3] - prev_box[1] + 1  # h

            #########apply box regression
            box_feat =  tfeats[max_idx].asnumpy()[:25088].copy()
            p_x = regr_x.decision_function(box_feat)
            p_y = regr_y.decision_function(box_feat)
            p_w = regr_w.decision_function(box_feat)
            p_h = regr_h.decision_function(box_feat)
            # print p_x, p_y, p_w, p_h
            new_x = p_x * prev_box[2] + prev_box[0] + 0.5 * prev_box[2]
            new_y = p_y * prev_box[3] + prev_box[1] + 0.5 * prev_box[3]
            new_w = prev_box[2] * np.exp(p_w)
            new_h = prev_box[3] * np.exp(p_h)

            new_x = new_x - 0.5 * new_w
            new_y = new_y - 0.5 * new_h

            ###################
            maxboxes[counter + 1, 1] = max(new_x, 0)
            maxboxes[counter + 1, 2] = max(new_y, 0)
            maxboxes[counter + 1, 3] = min(new_w + new_x - 1, im_w)
            maxboxes[counter + 1, 4] = min(new_h + new_y - 1, im_h)

            counter += 1

        endtime = datetime.datetime.now()
        print '[%d] %s done in %f seconds' % (vid, videoname, (endtime - starttime).seconds)
        np.savetxt(savefile, maxboxes[:,1:], delimiter = ',', fmt='%f')