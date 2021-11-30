import caffe_pb2
import lmdb
import cv2
from dataset import resize_with_aspectratio, center_crop

def convert_calib_list(data_dir, fn, out_dir):
    output_height = output_width = 224
    with open(fn) as f:
        total_num = sum(1 for i in f) * 1.1
    env = lmdb.open(out_dir, map_size=output_height*output_width*3*total_num)
    with env.begin(write=True) as txn, open(fn) as lf:
        for line in lf:
            line = line.strip(' \n')
            img = cv2.imread(os.path.join(data_dir, line))
            cv2_interpol = cv2.INTER_AREA
            img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
            img = center_crop(img, output_height, output_width)
            datum = caffe_pb2.Datum()
            datum.height, datum.width, datum.channels = img.shape
            datum.data = img.tobytes()
            txn.put(line.encode(), datum.SerializeToString())

if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 4:
        print('{} <dataset_dir> <calib_list.txt> <lmdb_dir>'.format(sys.argv[0]))
        sys.exit(1)
    convert_calib_list(*sys.argv[1:])

