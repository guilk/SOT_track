import os
import cv2
import cv
import numpy as np


if __name__ == '__main__':
    flow_root = '/data01/SOT/flows'
    img_root = '/data01/SOT/imgs'
    video_root = '/data01/SOT/videos'
    # flow_root = './flows'
    # img_root = './imgs'
    # video_root = './'

    # videos = ['Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', 'Coke', 'Couple', 'Crossing', 'David2', 'David3', 'David', 'Deer', 'Dog1', 'Doll', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Football', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Ironman', 'Jogging', 'Jumping', 'Lemming', 'Liquor', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Trellis', 'Walking2', 'Walking', 'Woman']
    # videos = ['Jogging']
    # videos = ['Freeman4']
    videos = ['Freeman4', 'MotorRolling', 'Suv', 'Jogging', 'Basketball', 'CarDark', 'FleetFace']
    for seq_name in videos:
        print seq_name

        flow_seq_path = os.path.join(flow_root, seq_name)
        img_seq_path = os.path.join(img_root, seq_name)



        video_path = os.path.join(video_root, seq_name+'.avi')
        # print video_path
        imgs = os.listdir(img_seq_path)
        imgs.sort()

        img_path = os.path.join(img_seq_path, imgs[0])
        tmp_img = cv2.imread(img_path)
        height, width, depth = tmp_img.shape


        video = cv2.VideoWriter(video_path, cv.CV_FOURCC('M','J','P','G'), 25, (width*2, height))

        for img_name in imgs:
            img_path = os.path.join(img_seq_path, img_name)
            # print img_path
            img = cv2.imread(img_path)
            flow_path = os.path.join(flow_seq_path, img_name)
            flow = cv2.imread(flow_path)
            conc = np.concatenate((img, flow), axis=1)
            # print conc.shape
            video.write(conc)
            # cv2.imshow('image', conc)
            # cv2.waitKey(10)
        video.release()

