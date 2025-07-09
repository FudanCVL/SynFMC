import math
from math import cos,sin
import numpy as np
import os
from PIL import Image
def norm(v1):
    return math.sqrt(sum([i**2 for i in v1]))

def normalize(vec):
    _norm=norm(vec)
    return [i/(_norm+1e-6) for i in vec]

def dot(v1,v2):
    return sum([a*b for a,b in zip(v1,v2)])



def get_degree(v1,v2):
    return math.degrees(math.acos(dot(v1,v2)/(norm(v1)*norm(v2))))



def cross(v1, v2):
    assert len(v1)==len(v2)
    result = [
        v1[1] * v2[2] - v1[2] * v2[1],  # x
        v1[2] * v2[0] - v1[0] * v2[2],  # y
        v1[0] * v2[1] - v1[1] * v2[0]   # z
    ]
    
    return result


def csv_param_to_dict(param, type_func, split=";"):
    values = param.split(split)
    dict_keys = []
    dict_values = []
    for value in values:
        if not value:
            continue
        dict_keys.append(value.split("=")[0])
        try:
            dict_values.append(type_func(value.split("=")[1]))
        except:
            dict_values.append(str(value.split("=")[1]))
    return dict(zip(dict_keys, dict_values))


def toradian(degree):
    return degree*(math.pi)/180

    

def transform_euler_to_matrix(roll, pitch, yaw):
    x,y,z=[toradian(i) for i in [roll,pitch,yaw]]

    return [
		[cos(y)*cos(z),-cos(y)*sin(z),-sin(y)],
		[sin(x)*sin(y)*cos(z)+cos(x)*sin(z),-sin(x)*sin(y)*sin(z)+cos(x)*cos(z),sin(x)*cos(y)],
		[cos(x)*sin(y)*cos(z)-sin(x)*sin(z),-cos(x)*sin(y)*cos(z)-sin(x)*cos(z),cos(x)*cos(y)],
	]
    

def transform_euler_to_matrix_v2(roll, pitch, yaw):
    x,y,z=[toradian(i) for i in [roll,pitch,yaw]]

    return [
		[cos(y)*cos(z),cos(z)*sin(x)*sin(y)-cos(x)*sin(z),-sin(x)*sin(z)-cos(x)*cos(z)*sin(y)],
		[cos(y)*sin(z),cos(x)*cos(z)+sin(x)*sin(y)*sin(z),-cos(x)*sin(z)*sin(y)+sin(x)*cos(z)],
		[sin(y),-cos(y)*sin(x),cos(x)*cos(y)],
	]
    


def is_mask_contained(big_mask,small_mask):
    
    assert small_mask is not None
    if big_mask is None:
        return False
        
    contains = np.all((big_mask & small_mask) == small_mask)
    
    return contains



def is_normal_size(
        mask,percentage=0.015
    ):  ## 1.5 %

        if len(mask.shape)==3:
            mask=mask[...,0]

        h, w = mask.shape[0], mask.shape[1]

        total_area = h * w

        rows, cols = np.where(mask >0 )
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            cur_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        else:
            cur_area = 0

        if cur_area > total_area * percentage:
            return True
        else:
            return False
            


def asseble_mask_list(mask_list):
    assert len(mask_list)>0
    
    total_mask=np.zeros_like(mask_list[0])
    for mask in mask_list:
        total_mask = total_mask | mask
    
    return total_mask

def get_present_continuous(verb):
    if verb.endswith('ie'):
        return verb[:-2] + 'ying'
    elif verb.endswith('e') and len(verb) > 2 and verb[-2] != 'e':
        return verb[:-1] + 'ing'
    elif (len(verb) >= 3 and
            verb[-1] not in 'aeiou' and
            verb[-2] in 'aeiou' and
            verb[-3] not in 'aeiou'):
        return verb + verb[-1] + 'ing'
    else:
        return verb + 'ing'


def get_third_person_singular(verb):
    if verb.endswith('y') and verb[-2] not in 'aeiou':
        return verb[:-1] + 'ies'
    elif verb.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
        return verb + 'es'
    else:
        return verb + 's'
    

import copy
import torch

def create_relative_matrix_of_cam_list(cam_info,scale_T = 1):
    
    RT_list = [np.copy(RT[:3].numpy()) for RT in cam_info]
    temp = []
    first_frame_RT = copy.deepcopy(RT_list[0])
    first_frame_R_inv = first_frame_RT[:,:3].T
    first_frame_R=first_frame_RT[:,:3]
    first_frame_T =  first_frame_RT[:,-1]
    for RT in RT_list:
        RT[:,-1] =  -np.dot(RT[:,:3].T,RT[:,-1]) + np.dot(RT[:,:3].T, first_frame_T)
        RT[:,:3] = np.dot(RT[:,:3].T, first_frame_R)
        RT[:,-1] = RT[:,-1] / scale_T
        temp.append(torch.from_numpy(RT))
    temp[0]= torch.eye(3, 4,dtype=temp[0].dtype)
    temp=[RT.reshape(-1) for RT in temp]
    return torch.stack(temp)



def create_absolute_matrix_from_ref_cam_list(first_cam_info,ref_cam_info_np,scale_T = 1):
    assert len(ref_cam_info_np)==16
    
    res=[np.copy(first_cam_info[:3])]

    
    for ref_cam_info in ref_cam_info_np[1:]:
        ref_cam_info=np.copy(ref_cam_info)        
        # Stack [0, 0, 0, 1] to the 3x4 ref_cam_info matrix
        ref_cam_info[:,-1]=ref_cam_info[:,-1] * scale_T
        bottom_row = np.array([0, 0, 0, 1])
        ref_cam_info = np.vstack((ref_cam_info, bottom_row))
        
        
        res.append(np.dot(first_cam_info,np.linalg.inv(ref_cam_info))[:3])
    
    return res

def create_relative_matrix_of_two_torch_matrix(RT1,RT2,scale_T = 1):
    # scale_T = 1
    # RT_list = [RT.reshape(3,4) for RT in RT_list]
    
    RT1 = np.copy(RT1[:3].numpy())
    RT2 = np.copy(RT2[:,:3].numpy())
    

    RT2[:,:,-1] =  -np.dot(RT2[:,:,:3].transpose((0, 2, 1)),RT2[:,:,-1:])[...,0,0] + np.dot(RT2[:,:,:3].transpose((0, 2, 1)), RT1[:,-1])
    RT2[:,:,:3] = np.dot(RT2[:,:,:3].transpose((0, 2, 1)), RT1[:,:3])
    RT2[:,:,-1] = RT2[:,:,-1] / scale_T

    # Flatten the last two dimensions of RT2
    RT2 = RT2.reshape(RT2.shape[0], -1)
    # RT2=RT2.reshape(-1)
    return RT2
