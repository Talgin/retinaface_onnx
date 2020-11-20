import os
import cv2
import numpy as np

import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnx import helper
from onnx import TensorProto
import onnxruntime.backend as backend

from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper

import time
import argparse

def resizeImg(img, dimension):
    dim = (dimension, dimension)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    #print(pred_boxes.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1]>4:
        pred_boxes[:,4:] = box_deltas[:,4:]

    return pred_boxes


def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
        pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
    return pred


def bbox_vote(det, nms_threshold):
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # nms
        merge_index = np.where(o >= nms_threshold)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                    axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def postprocess(net_out, threshold, ctx_id, im_scale, im_info):
    # im_info = [640, 640]
    flip = False
    decay4 = 0.5
    vote = False
    fpn_keys = []
    anchor_cfg = None
    bbox_stds = [1.0, 1.0, 1.0, 1.0]
    # im_scale = 1.0
    landmark_std = 1.0
    nms_threshold=0.4

    proposals_list = []
    scores_list = []
    landmarks_list = []
    strides_list = []

    use_landmarks = True

    if ctx_id>=0:
        nms = gpu_nms_wrapper(nms_threshold, ctx_id)
    else:
        nms = cpu_nms_wrapper(nms_threshold)

    use_landmarks = True
    _ratio = (1.,)

    _feat_stride_fpn = [32, 16, 8]
    anchor_cfg = {
          '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
        }

    for s in _feat_stride_fpn:
        fpn_keys.append('stride%s'%s)

    dense_anchor = False

    _anchors_fpn = dict(zip(fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=anchor_cfg)))
    for k in _anchors_fpn:
        v = _anchors_fpn[k].astype(np.float32)
        _anchors_fpn[k] = v

    _num_anchors = dict(zip(fpn_keys, [anchors.shape[0] for anchors in _anchors_fpn.values()]))
    sym_idx = 0

    for _idx,s in enumerate(_feat_stride_fpn):
        # print(sym_idx)
        _key = 'stride%s'%s
        # print(_key)
        stride = int(s)
        scores = net_out[sym_idx] #.asnumpy()

        scores = scores[:, _num_anchors['stride%s'%s]:, :, :]
        bbox_deltas = net_out[sym_idx+1] # .asnumpy()

        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = anchors_plane(height, width, stride, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)

        proposals = clip_boxes(proposals, im_info[:2])

        if stride==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()

        order = np.where(scores_ravel>=threshold)[0]

        proposals = proposals[order, :]
        scores = scores[order]
        if flip:
            oldx1 = proposals[:, 0].copy()
            oldx2 = proposals[:, 2].copy()
            proposals[:, 0] = im.shape[1] - oldx2 - 1
            proposals[:, 2] = im.shape[1] - oldx1 - 1

        #proposals[:,0:4] /= im_scale

        #print(proposals[:,0])
        proposals[:,0] /= im_scale[0]
        #print(pp)
        proposals[:,1] /= im_scale[1]
        proposals[:,2] /= im_scale[0]
        proposals[:,3] /= im_scale[1]
        #print(proposals[:,0])

        proposals_list.append(proposals)
        scores_list.append(scores)
        if nms_threshold<0.0:
            _strides = np.empty(shape=(scores.shape), dtype=np.float32)
            _strides.fill(stride)
            strides_list.append(_strides)
        if not vote and use_landmarks:
            landmark_deltas = net_out[sym_idx+2] #.asnumpy()
            # print(landmark_deltas)
            landmark_pred_len = landmark_deltas.shape[1]//A
            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
            landmark_deltas *= landmark_std
            landmarks = landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            if flip:
                landmarks[:,:,0] = im.shape[1] - landmarks[:,:,0] - 1
                order = [1,0,2,4,3]
                flandmarks = landmarks.copy()
                for idx, a in enumerate(order):
                    flandmarks[:,idx,:] = landmarks[:,a,:]
                landmarks = flandmarks
            landmarks[:,:,0:2] /= im_scale
            landmarks_list.append(landmarks)

        if use_landmarks:
            sym_idx += 3
        else:
            sym_idx += 2

    proposals = np.vstack(proposals_list)

    landmarks = None
    if proposals.shape[0]==0:
        if use_landmarks:
            landmarks = np.zeros( (0,5,2) )
        if nms_threshold<0.0:
            return np.zeros( (0,6) ), landmarks
        else:
            return np.zeros( (0,5) ), landmarks

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()

    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]

    scores = scores[order]
    if nms_threshold<0.0:
        strides = np.vstack(strides_list)
        strides = strides[order]
    if not vote and use_landmarks:
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

    if nms_threshold>0.0:
        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
        if not vote:
            keep = nms(pre_det)
            det = np.hstack( (pre_det, proposals[:,4:]) )
            det = det[keep, :]
            if use_landmarks:
                landmarks = landmarks[keep]
            else:
                det = np.hstack( (pre_det, proposals[:,4:]) )
                det = bbox_vote(det, nms_threshold)
    elif nms_threshold<0.0:
        det = np.hstack((proposals[:,0:4], scores, strides)).astype(np.float32, copy=False)
    else:
        det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    return det, landmarks


# onnx detection model inference
def onnx_model_detection(onnx_file, image_path, scales, resized_path):
    img = cv2.imread(image_path)
    print('Original shape:', img.shape)
    scales = scales

    im_shape = img.shape
    #target_size = scales[0]
    #max_size = scales[1]
    #im_size_x = im_shape[1]
    #im_size_y = im_shape[0]
    #im_scale_x = float(target_size) / float(im_size_x)
    #im_scale_y = float(target_size) / float(im_size_y)
    #if np.round(scales[0] * im_size_x) > max_size:
    #    im_scale_x = float(max_size) / float(im_size_x)
    #    im_scale_y = float(max_size) / float(im_size_y)
    #scales = [im_scale_x, im_scale_y]
    target_size = min(scales)
    max_size = max(scales)
    im_size_x = im_shape[1]
    im_size_y = im_shape[0]
    im_scale_x = float(scales[1]/im_size_x)
    im_scale_y = float(scales[0]/im_size_y)
    #if np.round(im_scale_x*im_size_x) > max_size:
    #    im_scale_x = float(im_size_x) / float(im_shape_x)
    #    im_scale_y = float(im_size_y) / float(im_shape_y)
    # if np.round(im_scale_y*im_size_y) > max
    scales = [im_scale_x, im_scale_y]
    print('im_scale:',scales)

    resized_img = None
    if im_scale_x!=1.0 or im_scale_y!=1.0:
        resized_img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(resized_path, resized_img)
    else:
        resized_img = img.copy()

    # if (img.shape[0] != 640) or img.shape[1] != 640:
    # if im_scale!=1.0:
    #     result_img = resizeImg(img, 640)
    #     cv2.imwrite('resized.jpg', result_img)
    # else:
    #     result_img = img
    # print('Resized shape:', result_img.shape)
    print('Resized shape:', resized_img.shape)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    aligned = np.transpose(resized_img, (2,0,1)) #HWC->CHW

    return aligned, img, scales


def get_feature(aligned, batching = False):
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
    ''' You can delete this one if not works '''
    # iterate through inputs of the graph
    # for input in onnx_model.graph.input:
    #     print (input.name, end=": ")
    #     # get type of input tensor
    #     tensor_type = input.type.tensor_type
    #     # check if it has a shape:
    #     if (tensor_type.HasField("shape")):
    #         # iterate through dimensions of the shape:
    #         for d in tensor_type.shape.dim:
    #             # the dimension may have a definite (integer) value or a symbolic identifier or neither:
    #             if (d.HasField("dim_value")):
    #                 print (d.dim_value, end=", ")  # known dimension
    #             elif (d.HasField("dim_param")):
    #                 print (d.dim_param, end=", ")  # unknown dimension with symbolic name
    #             else:
    #                 print ("?", end=", ")  # unknown dimension with no name
    #     else:
    #         print ("unknown rank", end="")
    #     print()
    ''' You can delete this one if not works '''

    ort_session = backend.prepare(onnx_model, 'GPU')
    #ort_session.get_providers()
    #ort.session.set_providers(['CUDAExecutionProvider'])
    outputs = ort_session.run(input_blob)
    # cnt = 0
    # print(len(outputs))
    #for output in outputs:
    #    print('Output '+str(cnt), output.shape)
    #    cnt+=1
    # print('im_scale:',scales)
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
            align, result_img, im_scale = onnx_model_detection(onnx_file, image_path, scales, img_name)
            aligned.append(align)
            results.append(result_img)
            names.append(image.split('.')[0])
            scales_lst.append(im_scale)

            if cnt != 0 and cnt%batching==0:
                outputs = get_feature(aligned, True)
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
            align, result_img, im_scale = onnx_model_detection(onnx_file, image_path, scales, img_name)
            outputs = get_feature(align)
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
