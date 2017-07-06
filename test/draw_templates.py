import os
import glob
import cv2
import cv
import numpy as np
import cPickle as pickle

if __name__ == '__main__':
    img_root = '/data01/SOT/imgs'
    video_root = '/data01/SOT/videos'

    videos = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', 'Coke', 'Couple', 'Crossing', 'David2', 'David3', 'David', 'Deer', 'Dog1', 'Doll', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping', 'Lemming', 'Liquor', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking2', 'Walking', 'Woman']
    # videos = ['Jogging']
    # videos = ['Car4']
    # videos = ['Freeman4', 'MotorRolling', 'Suv', 'Jogging', 'Basketball', 'CarDark', 'FleetFace']


    for seq_name in videos:
        print seq_name
        img_seq_path = os.path.join(img_root, seq_name,'*.jpg')
        img_paths = glob.glob(img_seq_path)
        tmp_img = cv2.imread(img_paths[0])
        height, width, depth = tmp_img.shape

        img_paths.sort()
        video_path = os.path.join(video_root, seq_name + '.avi')
        max_width = 0
        for img_path in img_paths:
            templ_path = img_path.replace('.jpg', '_template.pkl')
            infile = open(templ_path, 'rb')
            # with open(templ_path, 'rb') as infile:
            templates = pickle.load(infile)
            infile.close()
            total_width = 0
            for template in templates:
                total_width += template.shape[1]
            max_width = max(max_width, total_width)
        # print max_width
        # assert False
        video = cv2.VideoWriter(video_path, cv.CV_FOURCC('M', 'J', 'P', 'G'), 25, (width+max_width, height))

        for img_path in img_paths:
            templ_path = img_path.replace('.jpg', '_template.pkl')
            infile = open(templ_path, 'rb')
            # with open(templ_path, 'rb') as infile:
            templates = pickle.load(infile)
            infile.close()
            img = cv2.imread(img_path)
            # templates = np.load(templ_path)
            img_templ = np.zeros((height, max_width, depth))
            last_width = 0
            for template in templates:
                if template.ndim == 2:
                    template = template[:,:,np.newaxis]
                    template = np.tile(template,(1,1,3))

                img_templ[0:template.shape[0],last_width:last_width+template.shape[1]] = template
                last_width += template.shape[1]
            img_templ = img_templ.astype(np.uint8)
            # print img_templ.dtype
            # print img.dtype
            # print img.shape
            # print img_templ.shape
            conc = np.concatenate((img, img_templ), axis=1)
            video.write(conc)
        video.release()

