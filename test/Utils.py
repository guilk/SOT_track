
import numpy as np
import scipy.io as sio
import argparse
import image_io

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='eval_OTB')

    parser.add_argument('--overlapthresh', dest='ov_thresh',
                        help='overlap threshold to select training samples for box regression', default=0.6, type=float)
    parser.add_argument('--numangles', dest='nr_angles', help='num of angles for angular sampling for box regression',
                        default=20, type=int)
    parser.add_argument('--topk', dest='topk', help='num of top samples for flow checking', default=5, type=int)
    parser.add_argument('--flowtheta', dest='theta', help='threshold for flow inconsistency checking', default=0.25,
                        type=float)

    args = parser.parse_args()
    return args

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

def optical_flow_filter(flowfile, topk, theta, samples, prev_box, scores):

    rank = np.argsort(scores)
    k = topk
    candidates = rank[-k:]
    ##optical flow
    px1 = int(prev_box[0])
    py1 = int(prev_box[1])
    px2 = int(prev_box[2] + prev_box[0] - 1)
    py2 = int(prev_box[3] + prev_box[1] - 1)

    # print prev_box
    # flowfile = '%s%s/%04d.mat' % (optical_flow_root, videoname, id)
    try:
        flow = sio.loadmat(flowfile)
        flowvalue = flow['flow'].copy()
        flow_x = flowvalue[:, :, 0]
        flow_y = flowvalue[:, :, 1]
        flow_x_region = flow_x[py1 - 1:py2, px1 - 1:px2].flatten()  # row first
        flow_y_region = flow_y[py1 - 1:py2, px1 - 1:px2].flatten()

        xx, yy = np.meshgrid(np.arange(px1, px2 + 1, 1), np.arange(py1, py2 + 1, 1))
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()

        xx_next = xx_flat + flow_x_region
        yy_next = yy_flat + flow_y_region

        pixel_count = np.zeros((k,))
        for ii in range(k):
            bb = samples[:, candidates[ii]].copy()  # x1 y1 x2 y2
            flags = xx_next >= bb[0]
            flags = np.logical_and(flags, xx_next <= bb[2])
            flags = np.logical_and(flags, yy_next >= bb[1])
            flags = np.logical_and(flags, yy_next <= bb[3])
            pixel_count[ii] = np.sum(flags)
            # cv2.rectangle(img_draw, (int(bb[0]), int(bb[1])),
            #               (int(bb[2]), int(bb[3])),(255, 0, 0), 2)
        passed = pixel_count > (prev_box[2] * prev_box[3] * theta)
        if np.sum(passed) == 0:
            max_idx = np.argmax(scores)
        else:
            candidates_left = candidates[passed]
            max_idx = candidates_left[np.argmax(scores[candidates_left])]
            # for ii in range(len(candidates_left)):
            #     bb = samples[:, candidates_left[ii]].copy()
            #     cv2.rectangle(img_draw, (int(bb[0]), int(bb[1])),
            #                   (int(bb[2]), int(bb[3])), (0, 255, 0), 2)
    except:
        print 'could not read flow file: %s' % flowfile
        max_idx = np.argmax(scores)
    return max_idx

def prepare_tracking(srcRoot, gtfile, videoname, imageSz):
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
    frameidx = range(2, nr_frames2track + 1)
    if videoname == 'David':
        frameidx = range(301, nr_frames2track + 300)
    if videoname == 'Tiger1':
        frameidx = range(7, nr_frames2track + 6)
    input_roi = np.zeros((1, 5))
    input_roi[0, 1] = init_box[0] * imageSz / firstframe.shape[1]
    input_roi[0, 2] = init_box[1] * imageSz / firstframe.shape[0]
    input_roi[0, 3] = (init_box[0] + init_box[2] - 1) * imageSz / firstframe.shape[1]
    input_roi[0, 4] = (init_box[1] + init_box[3] - 1) * imageSz / firstframe.shape[0]
    input_roi[0, 1:] -= 1  # starting from 0
    input_roi = input_roi[np.newaxis, :]

    return firstframe,input_roi,init_box,frameidx,nr_frames2track
