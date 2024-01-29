import glob
from tqdm import tqdm
import extract_utils as utils
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import pickle
import os
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans


def metric(original_matrix,typ):
    if typ!='dot':
        output_matrix = 1/(1+pairwise_distances(original_matrix, metric=typ))
        output_matrix = np.nan_to_num(output_matrix , nan=0.0, posinf=0.0, neginf=0.0)
        return output_matrix
    else:
        return original_matrix @ original_matrix.T
    

def calculate_entropy(probabilities):
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()

def torchentropy(emd,maxENT=5):
    eps=0.000001 
    e = emd.size(1)
    length = emd.size(0) * emd.size(1)    
    hist = []; entropy=[]
    for v in range(e):
        entropy.append(calculate_entropy((eps + torch.histc(emd[:, v].view(-1), bins=30)) / length))
    
    entropy = torch.nan_to_num(torch.tensor(entropy), nan=maxENT)
    return entropy
    
def extract_stabilized_feature(feats,dr=3):
        
    if dr!=1:
        entropy = torchentropy(feats)
        ln = int(len(entropy)//dr)
        v = np.argsort(entropy)[0:ln]
        feats = feats[:,v]
    
    feats = F.normalize(feats, p=2, dim=-1)

    return feats


def our_affinty(feats_sep,K=5):
    eps=0.000000001
    feats_sep = feats_sep.cpu().numpy()
    eigs={}
       
    W_featb = metric(feats_sep,'braycurtis')
    W_featc = metric(feats_sep,'chebyshev')
    W_feat = W_featb  / (W_featc)

    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()
    W_comb = W_feat
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]#

   
    return [eigenvalues, eigenvectors]


def extract_instance_eigs(feats_root,masks_root,export_root,std_threshold=60):
    
    feats_path = np.sort(glob.glob(feats_root+'*.pth'))
    
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    for path in tqdm(feats_path):
        
        name = path.split('/')[-1]
        
        out = export_root + name

        if os.path.exists(out)==False:
            
            bg = cv2.imread(masks_root+name.replace('pth','png'),0)
            bg[bg==255]=1;
           
            data_dict = torch.load(path,map_location='cpu')
            hp = data_dict['shape'][2]//16; wp=data_dict['shape'][3]//16;
            h = data_dict['shape'][2]; w=data_dict['shape'][3];
            nbg = bg.reshape(hp*wp,1)

            feats = data_dict['k'].squeeze()#.cuda()
            feats_out = extract_stabilized_feature(feats)

            #sort with std
            index = torch.argsort(torch.std(feats_out,dim=0),descending=True)[:std_threshold]
            feats_sep = feats_out[:,index]*torch.from_numpy(nbg)#.cuda()

            output  = our_affinty(feats_sep)
            
            with open(out, 'wb') as fp:
               pickle.dump(output, fp)

def sep(query):
    allmask=[]
    for u in np.unique(query):
        if u!=0:
            mask = np.zeros_like(query)
            tp = np.where(query==u)
            mask[tp]=1;
            allmask.append(mask)
    return allmask

def get_infomask(pred_root,gt_root):
    
    preds_path = np.sort(glob.glob(pred_root+'*.png'))
    pred_masks={};gt_masks={}
    for path in tqdm(preds_path):
        
        name = path.split('/')[-1]
        pred_mask = cv2.imread(path,0)
        pred_mask[pred_mask!=0]=1;
        hp,wp = pred_mask.shape
        
        gt_mask = cv2.imread(gt_root+name.replace('_','/'),0)
        gt_mask =  cv2.resize(gt_mask, (wp, hp), interpolation=cv2.INTER_NEAREST)
        gt_mask = sep(gt_mask)
        
        pred_masks.update({name.replace('.png',''):pred_mask})
        gt_masks.update({name.replace('.png',''):gt_mask})
   
    return pred_masks,gt_masks

def core(Embedding,fgbg,K):
    
    yps = np.where(fgbg!=0);
    foreground_pixels = Embedding[fgbg > 0].reshape(-1, Embedding.shape[2])

    # Perform k-means clustering with K=2 on the foreground pixels
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(foreground_pixels)
    labels = kmeans.labels_
    
    # Create a new image where each pixel is colored based on its cluster assignment
    clustered_image = np.zeros_like(fgbg)
    clustered_image[yps] = 1+labels

    #prediction_masks = sep(clustered_image)
    #mean_iou,precisionT,recallT,f1T,accuracyT = need.calculate_instance_segmentation_accuracy(gt_mask, prediction_masks)
    return clustered_image

            
def clustring(eigs_root,fgbg_root,gt_root,export_root):
    
    fgbg_masks,gts_masks = get_infomask(fgbg_root,gt_root)
    eigs_path = np.sort(glob.glob(eigs_root+'*.pth'))
        
    if not os.path.exists(export_root):
        os.makedirs(export_root)
        
    for path in tqdm(eigs_path):
        
        name = path.split('/')[-1].replace('.pth','')
        out = export_root + name+'.png'
        
        with open(path, 'rb') as fp:
            file = pickle.load(fp)

        fgbg = fgbg_masks[name]; gt_mask=gts_masks[name]; hp,wp=fgbg.shape
        number_instances = len(gt_mask)

        [eigenvalues,eigenvectors] = file

        Embedding = eigenvectors[1:].permute(1,0).reshape(hp,wp,4).numpy()
        for d in range(0,Embedding.shape[2]):
            tmp = cv2.medianBlur(Embedding[:,:,d], 5)*fgbg
            Embedding[:,:,d] = tmp

        clustered_image=core(Embedding,fgbg,K=number_instances)
        
        cv2.imwrite(out,clustered_image*50)       
        
    
    
    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code')
    parser.add_argument('--type', help='select extract_eigs or clustring')
    parser.add_argument('--feats_root', help='feature root path')
    parser.add_argument('--fgbg_root', help='prediction mask fgbg root path')
    parser.add_argument('--gt_root', help='grand-truth mask root path')
    parser.add_argument('--eigs_root', help='eigs root path')
    parser.add_argument('--export_root', help='export root path')
    parser.add_argument('--std_threshold', type=int, help='std threshold')
    
    args = parser.parse_args()
    if args.type=='extract_eigs':
        extract_instance_eigs(args.feats_root,args.fgbg_root,args.export_root,args.std_threshold)
    else:
        clustring(args.eigs_root,args.fgbg_root,args.gt_root,args.export_root)
        
    