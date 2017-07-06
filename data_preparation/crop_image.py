import os
import numpy as np
import cv2


if __name__ == '__main__':
    print 'Crop image from ALOV dataset to fit training data'

    db_root = '/data01/ALOV/'
    img_root = '/data01/ALOV/images/'
    dst_root = '/data01/ALOV/cropped_images'
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    ann_root = '/data01/ALOV/annotations/'
    seq_types = os.listdir(img_root)
    seq_types.sort()
    for seq_type in seq_types:
        video_root = os.path.join(ann_root, seq_type)
        videos = os.listdir(video_root)
        videos.sort()
        for video in videos:
            ann_path = os.path.join(video_root, video)
            anns = np.loadtxt(ann_path,delimiter=' ',dtype=np.float32)
            dst_folder = os.path.join(dst_root, seq_type, video.split('.')[0])
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            assert anns.shape[1] == 9 # 00000105.jpg
            for index in range(anns.shape[0]):
                img_name = str(int(anns[index,0])).zfill(8)+'.jpg'
                info_name = str(int(anns[index,0])).zfill(8)+'.txt'
                dst_img_path = os.path.join(dst_folder, img_name)
                print dst_img_path
                dst_txt_path = os.path.join(dst_folder, info_name)

                x = np.min(anns[index,1::2])
                y = np.min(anns[index,2::2])
                w = np.max(anns[index,1::2]) - np.min(anns[index,1::2]) + 1
                h = np.max(anns[index,2::2]) - np.min(anns[index,2::2]) + 1

                img_path = os.path.join(img_root, seq_type, video.split('.')[0], img_name)
                im = cv2.imread(img_path)
                im_h = im.shape[0]
                im_w = im.shape[1]
                # print x, y, w, h, im_w, im_h
                max_pad = 1.1 * max(w,h)
                center_x = x + w/2.0
                center_y = y + h/2.0
                x1 = max(0, center_x - max_pad - w/2.0)
                x2 = min(im_w, center_x + max_pad + w/2.0)
                y1 = max(0, center_y - max_pad - h/2.0)
                y2 = min(im_h, center_y + max_pad + h/2.0)

                crop_img = im[int(y1):int(y2),int(x1):int(x2)]
                cv2.imwrite(dst_img_path, crop_img)
                np.savetxt(dst_txt_path,np.asarray([x1,y1,x2,y2]).reshape((1,4)),delimiter=',',fmt='%.4f')

                # print x1>0,y1>0,x2<im_w,y2<im_h
                assert os.path.exists(img_path)
