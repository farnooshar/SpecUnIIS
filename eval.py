
from sklearn.metrics import f1_score
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(mask1, mask2):
   
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def calculate_instance_segmentation_accuracy(gt_instances, prediction_masks):

    iou_matrix = np.zeros((len(prediction_masks), len(gt_instances)))

    for i in range(len(prediction_masks)):
        for j in range(len(gt_instances)):
            
            sm = calculate_iou(prediction_masks[i], gt_instances[j])
            iou_matrix[i, j] = sm
 
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    mean_iou = 0
    num_matches = 0
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:
            mean_iou += iou_matrix[i, j]
            num_matches += 1

    if num_matches > 0:
        mean_iou /= len(gt_instances) 
    else:
        mean_iou = 0

    return mean_iou

def sep(query):
    allmask=[]
    for u in np.unique(query):
        if u!=0:
            mask = np.zeros_like(query)
            tp = np.where(query==u)
            mask[tp]=1;
            allmask.append(mask)
    return allmask

def get_infoinstance(pred_root,gt_root):
    
    preds_path = np.sort(glob.glob(pred_root+'*.png'))
    pred_masks={};gt_masks={}
    for path in tqdm(preds_path):
        
        name = path.split('/')[-1]
        pred_mask = cv2.imread(path,0)
        hp,wp = pred_mask.shape
        pred_mask = sep(pred_mask)
        
        gt_mask = cv2.imread(gt_root+name.replace('_','/'),0)
        gt_mask =  cv2.resize(gt_mask, (wp, hp), interpolation=cv2.INTER_NEAREST)
        gt_mask = sep(gt_mask)
        
        pred_masks.update({name:pred_mask})
        gt_masks.update({name:gt_mask})
   
    return pred_masks,gt_masks

def instance(pred_root,gt_root):
    
    pred_masks,gts_masks = get_infoinstance(pred_root,gt_root)
    
    preds_path = np.sort(glob.glob(pred_root+'*.png'))
    q=0;m_iouT=0;
    
    for path in tqdm(preds_path):
        name = path.split('/')[-1]
        gt_mask=gts_masks[name]
        pred_mask = pred_masks[name]
        
        miou = calculate_instance_segmentation_accuracy(gt_mask, pred_mask)
        m_iouT = m_iouT + miou; 
        q+=1;
    
    print('mIOU is '+str(m_iouT/q))
    return m_iouT/q

def calculate_metrics(ground_truth, predicted_segmentation):
    # Ensure both images have the same dimensions
    assert ground_truth.shape == predicted_segmentation.shape, "Images must have the same dimensions"

    # Convert the images to binary (0s and 1s) if they are not already
    ground_truth_binary = np.asarray(ground_truth > 0, dtype=np.uint8)
    predicted_binary = np.asarray(predicted_segmentation > 0, dtype=np.uint8)

    # Flatten the binary images
    ground_truth_flat = ground_truth_binary.flatten()
    predicted_flat = predicted_binary.flatten()

    # Calculate metrics
    
    f_score = f1_score(ground_truth_flat, predicted_flat)

    return f_score


def fgbg(pred_root,gt_root):
    
    preds_path = np.sort(glob.glob(pred_root+'*.png'))
    q=0;f_scoreu=0;
    for path in tqdm(preds_path):
        
        name = path.split('/')[-1].replace('_','/')
        pred_mask = cv2.imread(path,0)
        pred_mask[pred_mask!=0]=1;
        hp,wp = pred_mask.shape
        
        gt_mask = cv2.imread(gt_root+name,0)
        gt_mask =  cv2.resize(gt_mask, (wp, hp), interpolation=cv2.INTER_NEAREST)
        gt_mask[gt_mask!=0]=1;

        f_score = calculate_metrics(gt_mask, pred_mask)
        f_scoreu = f_scoreu+f_score; 
        q+=1;
    
    print('f-score is '+str(f_scoreu/q))
    return f_scoreu/q
        
        

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code')
    
    parser.add_argument('--type', help='grand-truth mask root path')
    parser.add_argument('--pred_root', help='prediction mask root path')
    parser.add_argument('--gt_root', help='grand-truth mask root path')
    
    args = parser.parse_args()
    if args.type=='fgbg':
        fgbg(args.pred_root,args.gt_root)        
    else:
        instance(args.pred_root,args.gt_root)
    