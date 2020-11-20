import os
import cv2
import numpy as np

import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnx import helper
from onnx import TensorProto
import onnxruntime.backend as backend

from postprocessing import *

import time
import argparse

# onnx detection model inference
def face_preparation(image_path, scales, resized_path):
    img = cv2.imread(image_path)
    print('Original shape:', img.shape)
    scales = scales

    im_shape = img.shape
    target_size = min(scales)
    max_size = max(scales)
    im_size_x = im_shape[1]
    im_size_y = im_shape[0]
    im_scale_x = float(scales[1]/im_size_x)
    im_scale_y = float(scales[0]/im_size_y)
    scales = [im_scale_x, im_scale_y]
    print('im_scale:',scales)

    resized_img = None
    if im_scale_x!=1.0 or im_scale_y!=1.0:
        resized_img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(resized_path, resized_img)
    else:
        resized_img = img.copy()

    print('Resized shape:', resized_img.shape)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    aligned = np.transpose(resized_img, (2,0,1)) #HWC->CHW

    return aligned, img, scales


def get_feature(onnx_file, aligned, batching = False):
    if not batching:
        # ONE INPUT
        input_blob = np.expand_dims(aligned, axis=0).astype(np.float32) #NCHW
        # ONE INPUT
    else:
        # BATCHING
        input_blob = np.expand_dims(aligned, axis=0).astype(np.float32) #NCHW
        input_blob = np.squeeze(input_blob)
        # BATCHING
    onnx_model = onnx.load(onnx_file)
    ort_session = backend.prepare(onnx_model, 'GPU')
    outputs = ort_session.run(input_blob)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video stream catching test')
    parser.add_argument('--batching', default=0, type=int, help='Batch size. If 0 or not set, not using batch')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--images', default='./test_images/', help='Folder of images to be processed')
    parser.add_argument('--model', default='./model/900_900_dynamic_model.onnx', help='Folder of images to be processed')
    parser.add_argument('--scales', default='900,900', help='Size of the model input image')
    args = parser.parse_args()
    batching = args.batching
    ''' Models
    './model/dynamic_model.onnx' - 18.11.2020 best model  1024,1280-with dynamic batch size
    './model/new_retinaface_r50_v1.onnx' - Original Retinaface converted to onnx without batching
    './model/mnet025_fix_gamma_v1.onnx' - Retinaface with mnet 25 backbone
    './model/900_900_dynamic_model.onnx' - Retinaface with 900x900 image input and dynamic batch size
    '''
    onnx_file = args.model
    image_folder = args.images

    scales = args.scales 
    scales = scales.split(',')
    scales = [int(scales[0]), int(scales[1])]

    cnt = 0
    threshold = 0.8
    face_count = 0
    if batching > 0:
        names = []
        aligned = []
        results = []
        scales_lst = []
        results_folder = './batch_results/'
        resized_images = './batch_resized/'
        start = time.time()
        for image in os.listdir(image_folder):
            image_path = image_folder + image
            img_name = resized_images + image.split('.')[0] + '_resized.jpg'
            align, result_img, im_scale = face_preparation(image_path, scales, img_name)
            aligned.append(align)
            results.append(result_img)
            names.append(image.split('.')[0])
            scales_lst.append(im_scale)

            if (cnt != 0 and cnt%batching==0):
                outputs = get_feature(onnx_file, aligned, True)
                print('Length outputs[0]:', len(outputs[0]))
                print('Length outputs:', len(outputs))
                n = 0
                for j in range(0, len(outputs[0])):
                    out = []
                    for jj in range(0, len(outputs)):
                        out.append(np.expand_dims(outputs[jj][n], axis=0).astype(np.float32))
                    faces, landmarks = postprocess(out, threshold, 0, scales_lst[n], scales)
                    if faces.shape[0] > 0:
                        print('find', faces.shape[0], 'faces')
                        for i in range(faces.shape[0]):
                            box = faces[i].astype(np.int)
                            color = (0,0,255)
                            cv2.rectangle(results[n], (box[0], box[1]), (box[2], box[3]), color, 2)
                            # drawing facial landmarks on each face on the image
                            if landmarks is not None:
                                landmark5 = landmarks[i].astype(np.int)
                                for l in range(landmark5.shape[0]):
                                    color = (0,0,255)
                                    if l==0 or l==3:
                                        color = (0,255,0)
                                    cv2.circle(results[n], (landmark5[l][0], landmark5[l][1]), 1, color, 2)
                            # incrementing our face count
                            face_count += 1
                        # resulting frame with drawed faces
                        filename = results_folder + names[n] + '_res.jpg'
                        print('writing', filename)
                        cv2.imwrite(filename, results[n])
                        n += 1
                aligned = []
                results = []
                names = []
                scales_lst = []
                print(cnt)
            cnt += 1
        print('Batchning took:', time.time()-start)
        print('Amount of faces found:', face_count)
    else:
        results_folder = './results/'
        resized_images = './resized/'
        start = time.time()
        for image in os.listdir(image_folder):
            image_path = image_folder + image
            img_name = resized_images + image.split('.')[0] + '_resized.jpg'
            align, result_img, im_scale = face_preparation(image_path, scales, img_name)
            outputs = get_feature(onnx_file, align)
            print('Length outputs[0]:', len(outputs[0]))
            print('Length outputs:', len(outputs))
            # ---------- POSTPROCESSING ----------
            faces, landmarks = postprocess(outputs, threshold, 0, im_scale, scales)
            # ---------- POSTPROCESSING ----------

            if faces.shape[0] > 0:
                print('find', faces.shape[0], 'faces')
                for i in range(faces.shape[0]):
                    #print('score', faces[i][4])
                    box = faces[i].astype(np.int)
                    #color = (255,0,0)
                    color = (0,0,255)
                    cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    if landmarks is not None:
                        landmark5 = landmarks[i].astype(np.int)
                        #print(landmark.shape)
                        for l in range(landmark5.shape[0]):
                            color = (0,0,255)
                            if l==0 or l==3:
                                color = (0,255,0)
                            cv2.circle(result_img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
                    # incrementing our face count
                    face_count += 1
                filename = results_folder + image.split('.')[0] + '_res.jpg'
                print('writing', filename)
                cv2.imwrite(filename, result_img)
        print('Without batching took:', time.time()-start)
        print('Amount of faces found:', face_count)
    print('Finished.')
