from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

def extract_features(
    images_list: str,
    images_root: Optional[str],
    model_name: str,
    batch_size: int,
    output_dir: str,
    which_block: int = -1,
):
    """
    Extract features from a list of images.

    Example:
        python extract.py extract_features \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --output_dir "./data/VOC2012/features/dino_vits16" \
            --model_name dino_vits16 \
            --batch_size 1
    """

    # Output
    utils.make_output_dir(output_dir)

    # Models
    model_name = model_name.lower()
    model, val_transform, patch_size, num_heads = utils.get_model(model_name)

    # Add hook
    if 'dino' in model_name or 'mocov3' in model_name:
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    else:
        raise ValueError(model_name)

    # Dataset
    filenames = Path(images_list).read_text().splitlines()
    dataset = utils.ImagesDataset(filenames=filenames, images_root=images_root, transform=val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    print('Dataset size' ,len(dataset))
    print('Dataloader size',len(dataloader))
    
    

    # Prepare
    accelerator = Accelerator(fp16=True, cpu=False)
    # model, dataloader = accelerator.prepare(model, dataloader)
    model = model.to(accelerator.device)

    # Process
    pbar = tqdm(dataloader, desc='Processing')
    for i, (images, files, indices) in enumerate(pbar):
        output_dict = {}

        '''
        B, C, H, W = images.shape
        H = int(H*0.7); W = int(W*0.7)
        
        #print('1',images.shape)
        images = images.permute(2,3,1,0).detach().numpy()
        images = cv2.resize(images[:,:,:,0], (W, H), interpolation=cv2.INTER_NEAREST)
        #print('2',images.shape)
        images = torch.from_numpy(images)
        images = images.permute(2,0,1).unsqueeze(0)
        '''
        
        # Check if file already exists
        id = Path(files[0]).stem
        output_file = Path(output_dir) / f'{id}.pth'
        if output_file.is_file():
            pbar.write(f'Skipping existing file {str(output_file)}')
            continue

        # Reshape image
        P = patch_size
        B, C, H, W = images.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        # images = F.interpolate(images, size=(H_pad, W_pad), mode='bilinear')  # resize image
        images = images[:, :, :H_pad, :W_pad]
        images = images.to(accelerator.device)

        # Forward and collect features into output dict
        if 'dino' in model_name or 'mocov3' in model_name:
            # accelerator.unwrap_model(model).get_intermediate_layers(images)[0].squeeze(0)
            model.get_intermediate_layers(images)[0].squeeze(0)
            # output_dict['out'] = out
            output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            # output_dict['q'] = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
            output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
            # output_dict['v'] = output_qkv[2].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        else:
            raise ValueError(model_name)

        # Metadata
        output_dict['indices'] = indices[0]
        output_dict['file'] = files[0]
        output_dict['id'] = id
        output_dict['model_name'] = model_name
        output_dict['patch_size'] = patch_size
        output_dict['shape'] = (B, C, H, W)
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}

        # Save
        accelerator.save(output_dict, str(output_file))
        accelerator.wait_for_everyone()
    
    print('Saved features to,output_dir')



def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def bestent(emd,maxENT=5):

    e = emd.size(1); n = emd.size(0); hist=[]
    for v in range(0,e):
        hist.append(np.histogram(emd[:,v].cpu().numpy().ravel(), bins=30)[0])
    
    length = n * e

    entropy = [np.nan_to_num(calculate_entropy(h/length), nan=maxENT) for h in hist]
    
    return entropy


def _extract_eig(
    inp: Tuple[int, str], 
    K: int, 
    images_root: str,
    output_dir: str,
    which_matrix: str = 'laplacian',
    which_features: str = 'k',
    normalize: bool = True,
    lapnorm: bool = True,
    which_color_matrix: str = 'knn',
    threshold_at_zero: bool = True,
    image_downsample_factor: Optional[int] = None,
    image_color_lambda: float = 10,
    dr: float = 3 # What fraction of channels should be conserved based on entropy?
):
    index, features_file = inp
    

    # Load 
    data_dict = torch.load(features_file, map_location='cpu')
    image_id = data_dict['file'][:-4]

    # Load
    output_file = str(Path(output_dir) / f'{image_id}.pth')
    if Path(output_file).is_file():
        print('Skipping existing file',str(output_file))
        return  # skip because already generated

    # Load affinity matrix
    feats = data_dict[which_features].squeeze().cuda()
    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)


    if dr!=1:
        entropy = bestent(feats)
        ln = int(len(entropy)//dr)
        v = np.argsort(entropy)[0:ln]
        feats = feats[:,v]


    # Eigenvectors of affinity matrix
    if which_matrix == 'affinity_torch':
        W = feats @ feats.T
        if threshold_at_zero:
            W = (W * (W > 0))

        eigenvalues, eigenvectors = torch.eig(W, eigenvectors=True)
        eigenvalues = eigenvalues.cpu()
        eigenvectors = eigenvectors.cpu()

    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity_svd':        
        USV = torch.linalg.svd(feats, full_matrices=False)
        eigenvectors = USV[0][:, :K].T.to('cpu', non_blocking=True)
        eigenvalues = USV[1][:K].to('cpu', non_blocking=True)

    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity':
        W = (feats @ feats.T)
        if threshold_at_zero:
            W = (W * (W > 0))
        W = W.cpu().numpy()
        eigenvalues, eigenvectors = eigsh(W, which='LM', k=K)
        eigenvectors = torch.flip(torch.from_numpy(eigenvectors), dims=(-1,)).T

    # Eigenvectors of matting laplacian matrix
    elif which_matrix in ['matting_laplacian', 'laplacian']:

        # Get sizes
        B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
        if image_downsample_factor is None:
            image_downsample_factor = P
        H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

        # Upscale features to match the resolution
        if (H_patch, W_patch) != (H_pad_lr, W_pad_lr):
            feats = F.interpolate(
                feats.T.reshape(1, -1, H_patch, W_patch), 
                size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
            ).reshape(-1, H_pad_lr * W_pad_lr).T


        ### Feature affinities 
        W_feat = (feats @ feats.T)


        if threshold_at_zero:
            W_feat = (W_feat * (W_feat > 0))


        W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
        W_feat = W_feat.cpu().numpy()

        ### Color affinities 
        # If we are fusing with color affinites, then load the image and compute
        if image_color_lambda > 0:

            # Load image
            image_file = str(Path(images_root) / f'{image_id}.jpg')
            image_lr = Image.open(image_file).resize((W_pad_lr, H_pad_lr), Image.BILINEAR)
            image_lr = np.array(image_lr) / 255.

            # Color affinities (of type scipy.sparse.csr_matrix)
            if which_color_matrix == 'knn':
                W_lr = utils.knn_affinity(image_lr)
            elif which_color_matrix == 'rw':
                W_lr = utils.rw_affinity(image_lr)

            # Convert to dense numpy array
            W_color = np.array(W_lr.todense().astype(np.float32))

        else:

            # No color affinity
            W_color = 0

        # Combine
        W_comb = W_feat + W_color * image_color_lambda  # combination
        D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

        # Extract eigenvectors
        if lapnorm:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
        else:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM')
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM')
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]

    # Save dict
    output_dict = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}
    torch.save(output_dict, output_file)


def extract_eigs(
    images_root: str,
    features_dir: str,
    output_dir: str,
    which_matrix: str = 'laplacian',
    which_color_matrix: str = 'knn',
    which_features: str = 'k',
    normalize: bool = True,
    threshold_at_zero: bool = True,
    lapnorm: bool = True,
    K: int = 20,
    image_downsample_factor: Optional[int] = None,
    image_color_lambda: float = 0.0,
    multiprocessing: int = 0,
    dr: float = 3
):
    """
    Extracts eigenvalues from features.
    
    Example:
        python extract.py extract_eigs \
            --images_root "./data/VOC2012/images" \
            --features_dir "./data/VOC2012/features/dino_vits16" \
            --which_matrix "laplacian" \
            --output_dir "./data/VOC2012/eigs/laplacian" \
            --K 5
    """
    utils.make_output_dir(output_dir)
    kwargs = dict(K=K, which_matrix=which_matrix, which_features=which_features, which_color_matrix=which_color_matrix,
                 normalize=normalize, threshold_at_zero=threshold_at_zero, images_root=images_root, output_dir=output_dir, 
                 image_downsample_factor=image_downsample_factor, image_color_lambda=image_color_lambda, lapnorm=lapnorm,dr=dr)
    print(kwargs)
    fn = partial(_extract_eig, **kwargs)
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))
    utils.parallel_process(inputs, fn, multiprocessing)

def remove_small_area(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # Find the index of the contour with the largest area
    largest_contour_index = np.argmax(contour_areas)

    # Get the area of the largest contour
    largest_contour_area = contour_areas[largest_contour_index]
    
    try:
        # Calculate the ratio of each contour area to the area of the largest contour
        area_ratios = [area / largest_contour_area for area in contour_areas]

        # Create an empty image of the same size as the input image
        refined_image = np.zeros_like(image)

        # Iterate through the contours and their corresponding area ratios
        for contour, ratio in zip(contours, area_ratios):
            # If the ratio is greater than or equal to 0.1, draw the contour on the refined image
            if ratio >= 0.1:
                cv2.drawContours(refined_image, [contour], -1, 1, thickness=cv2.FILLED)

        return refined_image
    except:
        return image
    
def invert(image):
    h,w = image.shape
    border = [image[:5,:].ravel(), image[:,:5].ravel(), image[h-5:,:].ravel(), image[:,w-5:].ravel()]
    flag=0;
    for b in border:
        tp = np.where(b==1)[0]
        if len(tp)/len(b)>0.9:
            flag+=1;
    if flag>=2:
        return 1-image
    else:
        return image
    
def postprocessing(mask):
    mask = mask.astype('uint8')
    mask[mask!=0]=1;
    mask = invert(mask)
    mask = remove_small_area(mask)
    mask = cv2.medianBlur(mask, 3)
    mask[mask!=0]=255;
    return mask
    
def _extract_single_region_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    threshold: float,
    output_dir: str,
    post_processing: bool
):
    index, (feature_path, eigs_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['id'])
    output_file = str(Path(output_dir) / f'{id}.png')
    if Path(output_file).is_file():
        print('Skipping existing file',str(output_file))
        return  # skip because already generated

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    
    # Eigenvector
    eigenvector = data_dict['eigenvectors'][1].numpy()  # take smallest non-zero eigenvector
    segmap = (eigenvector > threshold).reshape(H_patch, W_patch)

    # Save dict
    if post_processing:
        Image.fromarray(postprocessing(segmap)).convert('L').save(output_file)
    else:
        Image.fromarray(segmap).convert('L').save(output_file)


def extract_single_region_segmentations(
    features_dir: str,
    eigs_dir: str,
    output_dir: str,
    threshold: float = 0.0,
    multiprocessing: int = 0,
    post_processing: bool = True
):
    """
    Example:
    python extract.py extract_single_region_segmentations \
        --features_dir "./data/VOC2012/features/dino_vits16" \
        --eigs_dir "./data/VOC2012/eigs/laplacian" \
        --output_dir "./data/VOC2012/single_region_segmentation/patches" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_single_region_segmentations, threshold=threshold, output_dir=output_dir,post_processing=post_processing)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir)
    utils.parallel_process(inputs, fn, multiprocessing)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        extract_features=extract_features,
        extract_eigs=extract_eigs,
        extract_single_region_segmentations=extract_single_region_segmentations,
    ))
