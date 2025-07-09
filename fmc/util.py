import torch
from einops import rearrange
from decord import VideoReader, cpu

import numpy as np
import random
from PIL import Image




UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def get_traj_features_v2(obj_info_list_list,obj_mask_list_list, omcm,cfg_random_null_om,cfg_random_null_om_ratio,is_cm_condition_null_list,local_rank,dtype):
    assert len(obj_info_list_list)==len(obj_mask_list_list)
    
    B=len(obj_info_list_list)
    F=len(obj_info_list_list[0])
    
    H,W=obj_mask_list_list[0][0].shape[-2],obj_mask_list_list[0][0].shape[-1]
    
    C1=12
    C2=1
    C=C1+C2
    traj_features=torch.zeros([B,F,H,W,C1],dtype=dtype).to(local_rank)
    mask_features=torch.zeros([B,F,H,W,C2],dtype=dtype).to(local_rank)

    for b_idx,(obj_info_list, obj_mask_list) in enumerate(zip(obj_info_list_list, obj_mask_list_list)):
        # processed_tensors = []
        for f_idx,(obj_info, obj_mask) in enumerate(zip(obj_info_list, obj_mask_list)):
            # obj_info shape: [obj_size, 12]
            # obj_mask shape: [obj_size, H, W]
            # if len(obj_mask.shape)==4:
            #     obj_mask=obj_mask[0]
            obj_size, c,H, W = obj_mask.shape ##c=1
            obj_mask=obj_mask.permute(0,2,3,1).to(device=local_rank,dtype=dtype)
            
            # Expand obj_info to [obj_size, H, W, 12]
            expanded_obj_info = torch.from_numpy(obj_info).unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
            # Convert expanded_obj_info to the same dtype as traj_features
            expanded_obj_info = expanded_obj_info.to(device=local_rank,dtype=dtype)
            # Apply the mask
            masked_obj_info = expanded_obj_info * obj_mask
            
            for _obj_info,_obj_mask in zip(masked_obj_info,obj_mask):
                __obj_mask=_obj_mask>0
                traj_features[b_idx][f_idx][__obj_mask[...,0]]= _obj_info[__obj_mask[...,0]]
                # mask_features[b_idx][f_idx][_obj_mask[...,0]]=1
                mask_features[b_idx][f_idx][__obj_mask[...,0]]=_obj_mask[__obj_mask[...,0]]
                # mask_features[b_idx][f_idx][__obj_mask[...,0]]=1
    
    for b in range(B):
        # Create a video tensor for this batch
        mask_video = mask_features[b]  # This is already in the format [F, H, W, C2]
        
        # Convert to numpy and uint8 for easier saving/visualization if needed
        mask_video_np = (mask_video.cpu().numpy() * 255).astype(np.uint8)
        
    features=torch.cat([traj_features,mask_features],dim=-1)  
    # features=mask_features
    if cfg_random_null_om:
        for i in range(features.shape[0]):
            # features[i] = features[i] if (random.random() > cfg_random_null_om_ratio and not is_cm_condition_null_list[i]) else torch.zeros_like(features[i])
            features[i] = features[i] if (random.random() > cfg_random_null_om_ratio) else torch.zeros_like(features[i])
    b, f, h, w ,c= features.shape
    
    features=features*mask_features
    features = rearrange(features, "b f h w c -> (b f) c h w").to(local_rank)
    
    mask_features=rearrange(mask_features, "b f h w c -> (b f) c h w").to(local_rank)
    traj_features = omcm(features,mask_features)
    # traj_features = omcm(features)
    traj_features = [rearrange(traj_feature, "(b f) c h w -> b c f h w", f=f) for traj_feature in traj_features] 
            
    # b, c, f, h, w = trajs.shape
    # trajs = rearrange(trajs, "b c f h w -> (b f) c h w")
    # traj_features = omcm(trajs)
    # traj_features = [rearrange(traj_feature, "(b f) c h w -> b c f h w", f=f) for traj_feature in traj_features]

    return traj_features



def get_traj_features(trajs, omcm):
    b, c, f, h, w = trajs.shape
    trajs = rearrange(trajs, "b c f h w -> (b f) c h w")
    traj_features = omcm(trajs)
    traj_features = [rearrange(traj_feature, "(b f) c h w -> b c f h w", f=f) for traj_feature in traj_features]

    return traj_features


def get_batch_motion(opt_model, num_reg_refine, batch_x, t=None):
    # batch_x: [-1, 1], [b, c, t, h, w]
    # input of motion model should be in range [0, 255]
    # output of motion model [B, 2, t, H, W]

    # batch_x [B, c, t, h, w]
    # b, c, t, h, w = batch_x.shape
    t = t if t is not None else batch_x.shape[2]
    batch_x = (batch_x + 1) * 0.5 * 255.

    motions = []
    for i in range(t-1):
        image1 = batch_x[:, :, i]
        image2 = batch_x[:, :, i+1]

        with torch.no_grad():
            results_dict = opt_model(image1, image2,
                                    attn_type='swin',
                                    attn_splits_list=[2, 8],
                                    corr_radius_list=[-1, 4],
                                    prop_radius_list=[-1, 1],
                                    num_reg_refine=num_reg_refine,
                                    task='flow',
                                    pred_bidir_flow=False,
                                    )
        motions.append(results_dict['flow_preds'][-1]) # [B, 2, H, W]

    motions = [torch.zeros_like(motions[0])] + motions # append a zero motion for the first frame
    motions = torch.stack(motions, dim=2) # [B, 2, t, H, W]

    return motions

def get_opt_from_video(opt_model, num_reg_refine, video_path, width, height, num_frames, device):
    video_reader = VideoReader(str(video_path), ctx=cpu(0),
                                   width=width, height=height)
    fps_ori = video_reader.get_avg_fps()
    frame_stride = len(video_reader) // num_frames
    frame_stride = min(frame_stride, 4)

    frame_indices = [frame_stride*i for i in range(num_frames)]
    video_data = video_reader.get_batch(frame_indices).asnumpy()
    video_data = torch.Tensor(video_data).permute(3, 0, 1, 2).float() # [c, t, h, w]
    video_data = video_data / 255.0 * 2.0 - 1.0
    # video_data = video_data.unsqueeze(0).repeat(2, 1, 1, 1, 1) # [2, c, t, h, w]
    video_data = video_data.unsqueeze(0) # [1, c, t, h, w]
    video_data = video_data.to(device)
    optcal_flow = get_batch_motion(opt_model, num_reg_refine, video_data, t=num_frames)

    return optcal_flow

def vis_opt_flow(flow):
    # flow: [b c t h w]
    vis_flow = []

    for i in range(0, flow.shape[2]):
        cur_flow = flow[0, :, i].permute(1, 2, 0).data.cpu().numpy()
        cur_flow = flow_to_image(cur_flow)
        vis_flow.append(cur_flow)
    vis_flow = np.stack(vis_flow, axis=0)
    vis_flow = vis_flow[:, :, :, ::-1] # [t, h, w, c]
    vis_flow = vis_flow / 255.0 # [0, 1]
    # [t, h, w, c] -> [c, t, h, w]
    vis_flow = rearrange(vis_flow, "t h w c -> c t h w")
    vis_flow = torch.Tensor(vis_flow) # [c, t, h, w]
    vis_flow = vis_flow[None, ...]

    return vis_flow