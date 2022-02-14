import re
import os
import argparse
import shutil


def run(img_list, src_dir, dst_dir):
    with open(img_list) as f:
        lines = f.readlines()
        for l in lines:
            file_name = '/' + l.strip('\n')+ '.jpg'
            print(file_name)
            shutil.copyfile(src_dir + file_name, dst_dir + file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-list', default='/home/onager/source/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
    parser.add_argument('--img-dir', default='/home/onager/source/datasets/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument('--dst-dir', default='/home/user/bmnnsdk2-bm1684_v2.6.0/BMService/data/VOC2012/JPEGImages')
    args = parser.parse_args()
    run(args.img_list, args.img_dir, args.dst_dir)
