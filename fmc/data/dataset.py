import os
import random
import json
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver

import math
import csv
import cv2

from .utils import *
from copy import deepcopy
import imageio
from nltk.stem import WordNetLemmatizer,PorterStemmer
from einops import rearrange

def get_background_description(hdri_json_data,cam_seq_data):
    comment_dict=csv_param_to_dict(cam_seq_data["Comment"], str)
    scene_type=comment_dict["scene_type"]
    hdri_id=comment_dict["hdri"]
    
    descriptions=hdri_json_data[hdri_id].get("descriptions",[])
    if len(descriptions)==0:
        description=""
    else:
        description=random.choice(descriptions)
        
    if description=="":
        if scene_type=="near_ground":
            scene_type="near ground"
        description=scene_type
    
    return description
    




def get_seen_object_and_action_description_val(annotation_data,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.005,**kwargs):
    obj_description_list,action_description_list,action_type_list=[],[],[]
    # objs_dict[seen_obj_idx][time_idx][-3:]
    
    cam_seq_data=seq_meta_data["camera"]
    objs_seq_data=seq_meta_data["objects"]
    
    obj_num=len(objs_seq_data)
    
    obj_id_list=[]
    for obj_idx in range(obj_num):
        obj_seq_data=objs_seq_data[str(obj_idx)]
        obj_comment_dict=csv_param_to_dict(obj_seq_data["Comment"], str)
        obj_id_list.append(obj_comment_dict["obj_id"])
        
    seen_obj_id_list,seen_obj_idx_list=[],[]

    
    is_obj_in_image_list=[None for _ in range(obj_num)]
    
    obj_dict=annotation_data["objects"]
    
    mask_h,mask_w=kwargs.get("height"),kwargs.get("width")
    
    total_mask=np.zeros((mask_h,mask_w,1)).astype(bool)
    
    
    obj_mask_list=[]
    
    cam_info=annotation_data["camera"][time_idx]
    
    cam_xyz,cam_euler_rot=cam_info[:3],cam_info[3:6]
    cam_rot=transform_euler_to_matrix_v2(cam_euler_rot[2],cam_euler_rot[1],cam_euler_rot[0]) ##,roll, pitch , yaw
    # Create 4x4 transformation matrix for camera
    cam_RT = np.eye(4)
    cam_RT[:3, :3] = cam_rot
    cam_RT[:3, 3] = cam_xyz
    
    focal_lenth,sensor_width,sensor_height=cam_info[-4],cam_info[-3],cam_info[-2]    
    for obj_id in obj_dict:
        comment_dict=csv_param_to_dict(seq_meta_data["objects"][obj_id]["Comment"], str)
        sphere_radius=float(comment_dict["sphere_radius"])
        obj_info=obj_dict[obj_id][time_idx]
        obj_xyz,obj_euler_rot=obj_info[-3:],obj_info[3:6]
        obj_rot=transform_euler_to_matrix(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw 
        # Create 4x4 transformation matrix for object
        obj_RT = np.eye(4)
        obj_RT[:3, :3] = obj_rot
        obj_RT[:3, 3] = obj_xyz
        
        relative_matrix=create_relative_matrix_of_two_torch_matrix(obj_RT,cam_RT,scale_T = 1)
        
        # Apply the relative transformation to the origin point
        origin = np.array([0, 0, 0, 1])  # Homogeneous coordinates
        transformed_origin = relative_matrix @ origin
        
        # Calculate projection parameters
        aspect_ratio = sensor_width / sensor_height
        fov_x = 2 * np.arctan(sensor_width / (2 * focal_lenth))
        fov_y = 2 * np.arctan(sensor_height / (2 * focal_lenth))
        
        

        
        # Project the transformed origin to camera plane
        x, y, z = transformed_origin[:3]
        
        
        if z > 0:  # Check if the point is in front of the camera
            x_proj = (x / z) * (mask_w / 2) / np.tan(fov_x / 2) + mask_w / 2
            y_proj = (y / z) * (mask_h / 2) / np.tan(fov_y / 2) + mask_h / 2
            
            center=[x_proj,y_proj]
            
            circle_mask = np.zeros_like(total_mask).astype(np.uint8)
            # cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, 0)            
            # Calculate the projected radius
            # First, calculate the distance from camera to object center
            distance = np.linalg.norm(transformed_origin[:3])
            
            # Calculate the projected radius using similar triangles
            projected_radius = (sphere_radius / distance) * (mask_w / 2) / np.tan(fov_x / 2)
            
            # Ensure the radius is at least 1 pixel
            radius = max(int(projected_radius), 1)
            # Draw a filled circle instead of just the outline
            cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
            circle_mask=(circle_mask>0)
            
            
            
            obj_mask_list.append(circle_mask)
            
            total_mask = np.logical_or(total_mask, circle_mask)
            
    
    AREA_PERCENTAGE=appearance_percentage
    
    if obj_num==1:
        if is_normal_size(total_mask,percentage=AREA_PERCENTAGE):
            obj_mask_list.append(total_mask)
            seen_obj_id_list.append(obj_id_list[0])
            seen_obj_idx_list.append(0)
    
    else:
        for i in range(obj_num):
            obj_mask=obj_mask_list[i]
            obj_mask=total_mask*obj_mask
            
            save_root="temp_mask_motion"
            data_type=kwargs["data_type"]
            # label_data=self.dataset[idx]
            seq_id=kwargs["seq_id"]
            pre_mask_img = Image.fromarray(obj_mask[...,0].astype(np.uint8)*255)
            save_dir=os.path.join(save_root,data_type,seq_id)
            os.makedirs(save_dir,exist_ok=True)
            pre_mask_img.save(os.path.join(save_dir,f"pre_mask-{time_idx}-{i}.png"))
            
            obj_mask_list.append(obj_mask)
            
        for idx in range(obj_num):
            obj_mask=obj_mask_list[idx]
            if is_normal_size(obj_mask,percentage=AREA_PERCENTAGE):#percentage=0.015
                is_obj_in_image_list[idx]=True
                obj_mask_list[idx]=obj_mask
            else:
                is_obj_in_image_list[idx]=False
                obj_mask_list[idx]=[]
                
        if not any(is_obj_in_image_list):
            pass
        
        new_obj_mask_list=[]
        choose_max=False
        # choose_max=True
        
        seen_num=0
        for idx in range(len(is_obj_in_image_list)):
            if is_obj_in_image_list[idx]:
                seen_num+=1
                if kwargs["max_num"] is not None and seen_num>kwargs["max_num"]:
                    break
                seen_obj_id_list.append(obj_id_list[idx])
                seen_obj_idx_list.append(idx)
                
                if choose_max:
                    cur_mask=obj_mask_list[idx]
                    num_labels, labels = cv2.connectedComponents(cur_mask.astype(np.uint8), connectivity=8)

                    max_connected_component = np.zeros_like(cur_mask).astype(bool)

                    sizes = np.bincount(labels.ravel())

                    sizes[0] = 0

                    max_label = sizes.argmax()

                    max_connected_component[labels == max_label] = True
                    new_obj_mask_list.append(max_connected_component)
                    
                else:
                    new_obj_mask_list.append(obj_mask_list[idx])
        
        if len(new_obj_mask_list)==0:
            pass
        obj_mask_list=new_obj_mask_list  
            

    
    
    for _,seen_obj_id in enumerate(seen_obj_id_list):
        obj_description=asset_json_data[seen_obj_id]["description"]
        # time_range=cam_time_range_list[seg_idx]
        seen_obj_idx=obj_id_list.index(seen_obj_id)
        assert seen_obj_idx!=-1
        time_range_list=eval(objs_seq_data[str(seen_obj_idx)]["Time_Range_List"])
        
        obj_seg_idx=-1
        for seg_idx,time_range in enumerate(time_range_list):
            start,end=time_range
            if start<=time_idx and end>=time_idx:
                obj_seg_idx=seg_idx
                break
        assert obj_seg_idx!=-1
        
        obj_comment_dict=csv_param_to_dict(objs_seq_data[str(seen_obj_idx)]["Comment"], str)
        animation_name_list=eval(obj_comment_dict["animation_name_list"])
        _action_type_list=eval(obj_comment_dict["action_type_list"])
        
        animation_name=animation_name_list[obj_seg_idx]
        action_type=_action_type_list[obj_seg_idx]
        
        action_description=asset_json_data[seen_obj_id]["animation"][animation_name].get("description","")
        
        obj_description_list.append(obj_description)
        action_description_list.append(action_description)
        action_type_list.append(action_type)
        
    return seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,obj_description_list,action_description_list,action_type_list


def get_seen_object_and_action_description_v3(mask_root,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.005,**kwargs):
    obj_description_list,action_description_list,action_type_list=[],[],[]
    
    cam_seq_data=seq_meta_data["camera"]
    objs_seq_data=seq_meta_data["objects"]
    
    obj_num=len(objs_seq_data)
    
    obj_id_list=[]
    for obj_idx in range(obj_num):
        obj_seq_data=objs_seq_data[str(obj_idx)]
        obj_comment_dict=csv_param_to_dict(obj_seq_data["Comment"], str)
        obj_id_list.append(obj_comment_dict["obj_id"])
        
    seen_obj_id_list,seen_obj_idx_list=[],[]

    is_obj_in_image_list=[None for _ in range(obj_num)]
    
    obj_mask_list=[]
    num = 0
    
    total_mask_path = os.path.join(mask_root, "total.png")
    total_mask = Image.open(total_mask_path)#.convert('L')
    total_mask = np.array(total_mask)
    total_mask = (total_mask > 0).astype(bool)
    total_mask = total_mask[..., np.newaxis]  # Add third dimension
    
    AREA_PERCENTAGE=appearance_percentage
    
    if obj_num==1:
        if is_normal_size(total_mask,percentage=AREA_PERCENTAGE):
            obj_mask_list.append(total_mask)
            seen_obj_id_list.append(obj_id_list[0])
            seen_obj_idx_list.append(0)
    
    else:
        for i in range(obj_num):
            obj_mask_path = os.path.join(mask_root, f"{i}.png")
            obj_mask = Image.open(obj_mask_path)#.convert('L')
            obj_mask = np.array(obj_mask)
            obj_mask = (obj_mask > 0).astype(bool)
            obj_mask = obj_mask[..., np.newaxis]  # Add third dimension
            obj_mask=total_mask*obj_mask
            
            # save_root="temp_mask_motion"
            # data_type=kwargs["data_type"]
            # # label_data=self.dataset[idx]
            # seq_id=kwargs["seq_id"]
            # pre_mask_img = Image.fromarray(obj_mask[...,0].astype(np.uint8)*255)
            # save_dir=os.path.join(save_root,data_type,seq_id)
            # os.makedirs(save_dir,exist_ok=True)
            # pre_mask_img.save(os.path.join(save_dir,f"pre_mask-{time_idx}-{i}.png"))
            
            obj_mask_list.append(obj_mask)

        for idx in range(obj_num):
            obj_mask=obj_mask_list[idx]
            if is_normal_size(obj_mask,percentage=AREA_PERCENTAGE):#percentage=0.015
                is_obj_in_image_list[idx]=True
                obj_mask_list[idx]=obj_mask
            else:
                is_obj_in_image_list[idx]=False
                obj_mask_list[idx]=[]
                
        if not any(is_obj_in_image_list):
            pass
        
        new_obj_mask_list=[]
        choose_max=False
        # choose_max=True
        
        seen_num=0
        for idx in range(len(is_obj_in_image_list)):
            if is_obj_in_image_list[idx]:
                seen_num+=1
                if kwargs["max_num"] is not None and seen_num>kwargs["max_num"]:
                    break
                seen_obj_id_list.append(obj_id_list[idx])
                seen_obj_idx_list.append(idx)
                
                if choose_max:
                    cur_mask=obj_mask_list[idx]
                    num_labels, labels = cv2.connectedComponents(cur_mask.astype(np.uint8), connectivity=8)

                    max_connected_component = np.zeros_like(cur_mask).astype(bool)

                    sizes = np.bincount(labels.ravel())

                    sizes[0] = 0

                    max_label = sizes.argmax()

                    max_connected_component[labels == max_label] = True
                    new_obj_mask_list.append(max_connected_component)
                    
                else:
                    new_obj_mask_list.append(obj_mask_list[idx])
        
        # if len(new_obj_mask_list)==0:
        #     pass
        obj_mask_list=new_obj_mask_list  
            

    
    
    for _,seen_obj_id in enumerate(seen_obj_id_list):
        obj_description=asset_json_data[seen_obj_id]["description"]
        # time_range=cam_time_range_list[seg_idx]
        seen_obj_idx=obj_id_list.index(seen_obj_id)
        assert seen_obj_idx!=-1
        time_range_list=eval(objs_seq_data[str(seen_obj_idx)]["Time_Range_List"])
        
        obj_seg_idx=-1
        for seg_idx,time_range in enumerate(time_range_list):
            start,end=time_range
            if start<=time_idx and end>=time_idx:
                obj_seg_idx=seg_idx
                break
        assert obj_seg_idx!=-1
        
        obj_comment_dict=csv_param_to_dict(objs_seq_data[str(seen_obj_idx)]["Comment"], str)
        animation_name_list=eval(obj_comment_dict["animation_name_list"])
        _action_type_list=eval(obj_comment_dict["action_type_list"])
        
        animation_name=animation_name_list[obj_seg_idx]
        action_type=_action_type_list[obj_seg_idx]
        
        action_description=asset_json_data[seen_obj_id]["animation"][animation_name].get("description","")
        
        obj_description_list.append(obj_description)
        action_description_list.append(action_description)
        action_type_list.append(action_type)
        
    return seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,obj_description_list,action_description_list,action_type_list


def get_seen_object_and_action_description_v2(exr_file_path,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.005,**kwargs):
    obj_description_list,action_description_list,action_type_list=[],[],[]
    
    cam_seq_data=seq_meta_data["camera"]
    objs_seq_data=seq_meta_data["objects"]
    
    obj_num=len(objs_seq_data)
    
    obj_id_list=[]
    for obj_idx in range(obj_num):
        obj_seq_data=objs_seq_data[str(obj_idx)]
        obj_comment_dict=csv_param_to_dict(obj_seq_data["Comment"], str)
        obj_id_list.append(obj_comment_dict["obj_id"])
        
    seen_obj_id_list,seen_obj_idx_list=[],[]

    
    is_obj_in_image_list=[None for _ in range(obj_num)]
    
    obj_mask_list=[]
    num = 0
    total_mask = read_exr_r_channel(exr_file_path, f"ActorHitProxyMask0{num}")
    
    total_mask=(total_mask<1e-5)
    AREA_PERCENTAGE=appearance_percentage
    
    if obj_num==1:
        if is_normal_size(total_mask,percentage=AREA_PERCENTAGE):
            obj_mask_list.append(total_mask)
            seen_obj_id_list.append(obj_id_list[0])
            seen_obj_idx_list.append(0)
    
    else:
        for i in range(obj_num):
            obj_rgb = read_exr_rgb_channel(
                exr_file_path, f"FinalImageactor_{i:02}_object"
            )

            obj_mask = (obj_rgb.mean(-1) > 0)[...,None]
            obj_mask=total_mask*obj_mask
            
            mask_root="temp_mask_motion"
            data_type=kwargs["data_type"]
            # label_data=self.dataset[idx]
            seq_id=kwargs["seq_id"]
            pre_mask_img = Image.fromarray(obj_mask[...,0].astype(np.uint8)*255)
            save_dir=os.path.join(mask_root,data_type,seq_id)
            os.makedirs(save_dir,exist_ok=True)
            pre_mask_img.save(os.path.join(save_dir,f"pre_mask-{time_idx}-{i}.png"))
            
            # print(caption,img_path_list[frame_indice])
            # obj_uint_mask=obj_mask.astype(np.uint8) * 255
            
            num_labels, labels = cv2.connectedComponents(obj_mask.astype(np.uint8), connectivity=8)
            

            mask_list=[]
            for i in range(1,num_labels):            
                connected_component = np.zeros_like(obj_mask)
                
                connected_component[labels==i] = 1
                mask_list.append(connected_component.astype(bool))

            # obj_mask_img = Image.fromarray(obj_mask.astype(np.uint8) * 255)
            # obj_mask_img.save(f"temp_exr{num}.jpg")
            
            obj_mask_list.append(mask_list)
            
        # if exr_file_path=="/home/volume_shared/share_datasets/data_nvme/traj_dataset/Rendered_Traj_Results_multi/dynamic/3/exr/0054.exr":
        #     pass
        
        for idx in range(obj_num):
            ord_mask_list=obj_mask_list[idx]
            mask_list=[]
            
            for mask in ord_mask_list:
                add_mask=True
                for idx2,other_mask_list in enumerate(obj_mask_list):
                    if idx2==idx:
                        continue
                    for other_mask in other_mask_list:
                        if is_mask_contained(other_mask,mask):
                            add_mask=False
                            break
                    if add_mask==False:
                        break
                if add_mask:
                    mask_list.append(mask)
                    

            
            if not mask_list:
                is_obj_in_image_list[idx]=False
                obj_mask_list[idx]=[]
            else:
                assembled_mask=asseble_mask_list(mask_list)
                if is_normal_size(assembled_mask,percentage=AREA_PERCENTAGE):#percentage=0.015
                    is_obj_in_image_list[idx]=True
                    obj_mask_list[idx]=assembled_mask
                else:
                    is_obj_in_image_list[idx]=False
                    obj_mask_list[idx]=[]
                    
        if not any(is_obj_in_image_list):
            pass
        
        new_obj_mask_list=[]
        choose_max=False
        # choose_max=True
        
        seen_num=0
        for idx in range(len(is_obj_in_image_list)):
            if is_obj_in_image_list[idx]:
                seen_num+=1
                if kwargs["max_num"] is not None and seen_num>kwargs["max_num"]:
                    break
                seen_obj_id_list.append(obj_id_list[idx])
                seen_obj_idx_list.append(idx)
                
                if choose_max:
                    cur_mask=obj_mask_list[idx]
                    num_labels, labels = cv2.connectedComponents(cur_mask.astype(np.uint8), connectivity=8)

                    max_connected_component = np.zeros_like(cur_mask).astype(bool)

                    sizes = np.bincount(labels.ravel())

                    sizes[0] = 0

                    max_label = sizes.argmax()

                    max_connected_component[labels == max_label] = True
                    new_obj_mask_list.append(max_connected_component)
                    
                else:
                    new_obj_mask_list.append(obj_mask_list[idx])
        
        if len(new_obj_mask_list)==0:
            pass
        obj_mask_list=new_obj_mask_list  
            

    
    
    for _,seen_obj_id in enumerate(seen_obj_id_list):
        obj_description=asset_json_data[seen_obj_id]["description"]
        # time_range=cam_time_range_list[seg_idx]
        seen_obj_idx=obj_id_list.index(seen_obj_id)
        assert seen_obj_idx!=-1
        time_range_list=eval(objs_seq_data[str(seen_obj_idx)]["Time_Range_List"])
        
        obj_seg_idx=-1
        for seg_idx,time_range in enumerate(time_range_list):
            start,end=time_range
            if start<=time_idx and end>=time_idx:
                obj_seg_idx=seg_idx
                break
        assert obj_seg_idx!=-1
        
        obj_comment_dict=csv_param_to_dict(objs_seq_data[str(seen_obj_idx)]["Comment"], str)
        animation_name_list=eval(obj_comment_dict["animation_name_list"])
        _action_type_list=eval(obj_comment_dict["action_type_list"])
        
        animation_name=animation_name_list[obj_seg_idx]
        action_type=_action_type_list[obj_seg_idx]
        
        action_description=asset_json_data[seen_obj_id]["animation"][animation_name].get("description","")
        
        obj_description_list.append(obj_description)
        action_description_list.append(action_description)
        action_type_list.append(action_type)
        
    return seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,obj_description_list,action_description_list,action_type_list



def get_seen_object_and_action_description(exr_file_path,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.005,**kwargs):
    obj_description_list,action_description_list,action_type_list=[],[],[]
    
    cam_seq_data=seq_meta_data["camera"]
    objs_seq_data=seq_meta_data["objects"]
    
    obj_num=len(objs_seq_data)
    
    obj_id_list=[]
    for obj_idx in range(obj_num):
        obj_seq_data=objs_seq_data[str(obj_idx)]
        obj_comment_dict=csv_param_to_dict(obj_seq_data["Comment"], str)
        obj_id_list.append(obj_comment_dict["obj_id"])
        
    seen_obj_id_list,seen_obj_idx_list=[],[]

    is_obj_in_image_list=[None for _ in range(obj_num)]
    
    obj_mask_list=[]
    num = 0
    total_mask = read_exr_r_channel(exr_file_path, f"ActorHitProxyMask0{num}")
    
    total_mask=(total_mask<1e-5)
    AREA_PERCENTAGE=appearance_percentage
    
    if obj_num==1:
        if is_normal_size(total_mask,percentage=AREA_PERCENTAGE):
            obj_mask_list.append(total_mask)
            seen_obj_id_list.append(obj_id_list[0])
            seen_obj_idx_list.append(0)
    
    else:
        for i in range(obj_num):
            obj_rgb = read_exr_rgb_channel(
                exr_file_path, f"FinalImageactor_{i:02}_object"
            )

            obj_mask = (obj_rgb.mean(-1) > 0)[...,None]
            obj_mask=total_mask*obj_mask
            
            mask_root="temp_mask"
            data_type=kwargs["data_type"]

            seq_id=kwargs["seq_id"]
            pre_mask_img = Image.fromarray(obj_mask[...,0].astype(np.uint8)*255)
            save_dir=os.path.join(mask_root,data_type,seq_id)
            os.makedirs(save_dir,exist_ok=True)
            pre_mask_img.save(os.path.join(save_dir,f"pre_mask-{time_idx}-{i}.png"))
            

            
            num_labels, labels = cv2.connectedComponents(obj_mask.astype(np.uint8), connectivity=8)
            

            mask_list=[]
            for i in range(1,num_labels):            
                connected_component = np.zeros_like(obj_mask)

                connected_component[labels==i] = 1
                mask_list.append(connected_component.astype(bool))


            
            obj_mask_list.append(mask_list)
            

        for idx in range(obj_num):
            ord_mask_list=obj_mask_list[idx]
            mask_list=[]
            
            for mask in ord_mask_list:
                add_mask=True
                for idx2,other_mask_list in enumerate(obj_mask_list):
                    if idx2==idx:
                        continue
                    for other_mask in other_mask_list:
                        if is_mask_contained(other_mask,mask):
                            add_mask=False
                            break
                    if add_mask==False:
                        break
                if add_mask:
                    mask_list.append(mask)
                    

            
            if not mask_list:
                is_obj_in_image_list[idx]=False
                obj_mask_list[idx]=[]
            else:
                assembled_mask=asseble_mask_list(mask_list)
                if is_normal_size(assembled_mask,percentage=AREA_PERCENTAGE):#percentage=0.015
                    is_obj_in_image_list[idx]=True
                    obj_mask_list[idx]=assembled_mask
                else:
                    is_obj_in_image_list[idx]=False
                    obj_mask_list[idx]=[]
                    
        if not any(is_obj_in_image_list):
            pass
        
        new_obj_mask_list=[]
        choose_max=False
        # choose_max=True
        
        seen_num=0
        for idx in range(len(is_obj_in_image_list)):
            if is_obj_in_image_list[idx]:
                seen_num+=1
                # if kwargs["max_num"] is not None and seen_num>kwargs["max_num"]:
                #     break
                seen_obj_id_list.append(obj_id_list[idx])
                seen_obj_idx_list.append(idx)
                
                if choose_max:
                    cur_mask=obj_mask_list[idx]
                    num_labels, labels = cv2.connectedComponents(cur_mask.astype(np.uint8), connectivity=8)

                    max_connected_component = np.zeros_like(cur_mask).astype(bool)

                    sizes = np.bincount(labels.ravel())

                    sizes[0] = 0

                    max_label = sizes.argmax()

                    max_connected_component[labels == max_label] = True
                    new_obj_mask_list.append(max_connected_component)
                    
                else:
                    new_obj_mask_list.append(obj_mask_list[idx])
        
        if len(new_obj_mask_list)==0:
            pass
        obj_mask_list=new_obj_mask_list  
            

    
    
    for _,seen_obj_id in enumerate(seen_obj_id_list):
        obj_description=asset_json_data[seen_obj_id]["description"]
        # time_range=cam_time_range_list[seg_idx]
        seen_obj_idx=obj_id_list.index(seen_obj_id)
        assert seen_obj_idx!=-1
        time_range_list=eval(objs_seq_data[str(seen_obj_idx)]["Time_Range_List"])
        
        obj_seg_idx=-1
        for seg_idx,time_range in enumerate(time_range_list):
            start,end=time_range
            if start<=time_idx and end>=time_idx:
                obj_seg_idx=seg_idx
                break
        assert obj_seg_idx!=-1
        
        obj_comment_dict=csv_param_to_dict(objs_seq_data[str(seen_obj_idx)]["Comment"], str)
        animation_name_list=eval(obj_comment_dict["animation_name_list"])
        _action_type_list=eval(obj_comment_dict["action_type_list"])
        
        animation_name=animation_name_list[obj_seg_idx]
        action_type=_action_type_list[obj_seg_idx]
        
        action_description=asset_json_data[seen_obj_id]["animation"][animation_name].get("description","")
        
        obj_description_list.append(obj_description)
        action_description_list.append(action_description)
        action_type_list.append(action_type)
        
    return seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,obj_description_list,action_description_list,action_type_list
        
        


def get_camera_pose_description(annotation_data,seen_obj_idx_list,time_idx):
    
    cam_dict,objs_dict=annotation_data["camera"],annotation_data["objects"]
    
    cam_type_list=[]##,distance_type_list,height_type_list=[],[],[]
    
    for seen_obj_idx in seen_obj_idx_list:
        seen_obj_idx=str(seen_obj_idx)
        cam_xyz,cam_euler_rot=cam_dict[time_idx][:3],cam_dict[time_idx][3:6]
        # cam_rot=transform_euler_to_matrix(cam_euler_rot[2],cam_euler_rot[1],cam_euler_rot[0]) ##,roll, pitch , yaw
        obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
        obj_rot=transform_euler_to_matrix(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
    
        cam_type=get_cam_type(obj_xyz,obj_rot,cam_xyz,front_degree_limit=30,left_degree_limit=30,height_degree_limit=30)

        cam_type_list.append(cam_type)

    return cam_type_list


def get_camera_pose_description_v2(annotation_data,seen_obj_idx_list,time_idx):
    
    cam_dict,objs_dict=annotation_data["camera"],annotation_data["objects"]
    
    cam_type_list=[]##,distance_type_list,height_type_list=[],[],[]
    
    for seen_obj_idx in seen_obj_idx_list:
        seen_obj_idx=str(seen_obj_idx)
        cam_xyz,cam_euler_rot=cam_dict[time_idx][:3],cam_dict[time_idx][3:6]
        # cam_rot=transform_euler_to_matrix(cam_euler_rot[2],cam_euler_rot[1],cam_euler_rot[0]) ##,roll, pitch , yaw
        obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
        obj_rot=transform_euler_to_matrix_v2(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
    
        cam_type=get_cam_type_v2(obj_xyz,obj_rot,cam_xyz,front_degree_limit=30,left_degree_limit=30,height_degree_limit=30)

        cam_type_list.append(cam_type)

    return cam_type_list
    
def compute_distance(xy1,xy2):
    distance=0
    for x,y in zip(xy1,xy2):
        distance+=(x-y)**2
    return math.sqrt(distance)





def get_cam_type(obj_xyz,obj_rot,cam_xyz,front_degree_limit,left_degree_limit,height_degree_limit):
    obj_to_cam_xyz=[j-i for i,j in zip(obj_xyz,cam_xyz)]

    y_axis=[row[1] for row in obj_rot]
    z_axis=[row[-1] for row in obj_rot]
    # ref_xy=obj_to_cam_xyz[:2]

    degree=get_degree(y_axis,normalize(obj_to_cam_xyz))
    
    xy_distance=compute_distance(obj_xyz[:2],cam_xyz[:2])
    
    
    if abs(90-degree)<front_degree_limit:
        front_str=""
        
    else:
        if degree<90:
            is_front=True
            front_str="front"
        else:
            is_front=False
            front_str="back"

    if abs(degree)<left_degree_limit or abs(180-degree)<left_degree_limit:
        left_str=""
    else:
        if dot(cross(y_axis,obj_to_cam_xyz),z_axis)<0:
            is_left=True
            left_str="left"
        else:
            is_left=False
            left_str="right"

    z_degree=get_degree(z_axis,normalize(obj_to_cam_xyz))
    
    if abs(90-z_degree)<height_degree_limit:
        top_str=""
        
    else:
        if z_degree<90:
            is_top=True
            top_str="top"
        else:
            is_front=False
            top_str="down"

    
    return "_".join([front_str,left_str,top_str])
    
            


def get_cam_type_v2(obj_xyz,obj_rot,cam_xyz,front_degree_limit,left_degree_limit,height_degree_limit):
    obj_to_cam_xyz=[j-i for i,j in zip(obj_xyz,cam_xyz)]
    y_axis=[row[1] for row in obj_rot]
    z_axis=[row[-1] for row in obj_rot]
    # ref_xy=obj_to_cam_xyz[:2]

    degree=get_degree(y_axis,normalize(obj_to_cam_xyz))
    
    xy_distance=compute_distance(obj_xyz[:2],cam_xyz[:2])
    
    
    if abs(90-degree)<front_degree_limit:
        front_str=""
        
    else:
        if degree<90:
            is_front=True
            front_str="front"
        else:
            is_front=False
            front_str="back"
    if abs(degree)<left_degree_limit or abs(180-degree)<left_degree_limit:
        left_str=""
    else:
        if dot(cross(y_axis,obj_to_cam_xyz),z_axis)<0:
            is_left=True
            left_str="left"
        else:
            is_left=False
            left_str="right"

    z_degree=get_degree(z_axis,normalize(obj_to_cam_xyz))
    
    if abs(90-z_degree)<height_degree_limit:
        top_str=""
        
    else:
        if z_degree<90:
            is_top=True
            top_str="top"
        else:
            is_front=False
            top_str="down"

    
    return "_".join([front_str,left_str,top_str])





class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1).to(c2w)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker






class UnrealTrajLoraDataset(Dataset):
    
    GROUND_MOVE_WORD_LIST=[
        "move",
        
        "walk",
        "shift",
        "stroll",
        
        "run",
        "dash",
        "sprint",
    ]
    
    OVERWATER_MOVE_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "glide",
        "swim",
        # "sail",
    ]
    
    JUMP_WORD_LIST=[
        "jump",
        "leap",
    ]
    
    FLY_WORD_LIST=[
        "move",
        
        "shift",
        
        "fly",
        "soar",
        "glide",

    ]
    
    NEAR_GROUND_FLY_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "fly",        
        "glide",
        
    ]
    
    SWIM_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "swim",
        "dive",
        
    ]

    
    GROUND_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        # "pause",
        "halt",
        
        # "freeze",
        # "stand",
        # "linger"
    ]
    
    
    SKY_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "hover",
        "suspend",
        
    ]
    
    NEAR_GROUND_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "hover",
        "suspend",
    ]
    
    
    OVERWATER_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "suspend",
    ]
    
    UNDERWATER_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "suspend",
    ]
    

    A_THE_LIST=["a","the"]
    
    # CONCAT_LIST=["and",", and","while",", while","."]
    CONCAT_LIST=["and",", and","."]
    
    ADJ_OBJ_TEMPLATE=[
        "{a_the} {object_name} which is {action_name_ing}",
        "{a_the} {action_name_ing} {object_name}",
        "{a_the} {object_name}",
    ]
    
    NO_ADJ_OBJ_TEMPLATE=[
        "{a_the} {object_name} {action_name}",
        "{a_the} {object_name} is {action_name_ing}"
    ]
    
    
    
    ADJ_CAM_TEMPLATE=[ ##no present tense
        # "{object_sentence} from {view_point_sentence} view",

        "{object_sentence} is viewed from {view_point_sentence} side",

        "{object_sentence} is observed from the {view_point_sentence} perspective",
        
        "camera captures {object_sentence} as seen from {view_point_sentence} perspective",

        "{object_sentence} is seen from {view_point_sentence} side",

        "{object_sentence} is viewed from {view_point_sentence} viewpoint",

        "{object_sentence} is captured at the {view_point_sentence} angle",

        # "the camera's {view_point_sentence} shot shows the {object_sentence}"
    ]
    
    NO_ADJ_CAM_TEMPLATE=[
        # "{object_sentence} from {view_point_sentence} side",

        "viewed from {view_point_sentence} perspective, {object_sentence}",
        
        "observed from {view_point_sentence} view, {object_sentence}",
        
        "seen from {view_point_sentence} view, {object_sentence}",
    ]
    

    BACK_ASSEMBLE_SINGLE_TEMPLATE=[
        "in {background}, {first_sentence}",
        "with {background} behind, {first_sentence}",
        "against {background}, {first_sentence}",
        "in front of {background}, {first_sentence}",        
        "with {background} in the background, {first_sentence}",
        "with {background} backdrop, {first_sentence}",

        "{first_sentence} in {background}",
        "{first_sentence} with {background} behind",
        "{first_sentence} against {background}",
        "{first_sentence} in front of {background}",  
        "{first_sentence} with {background} in the background",
        "{first_sentence} with {background} backdrop",

        
    ]
    
    NO_BACK_ASSEMBLE_SINGLE_TEMPLATE=[
        "{first_sentence}",    
    ]

    
    NO_BACK_ASSEMBLE_MULTI_TEMPLATE=[        
        "{first_sentence} {concat} {second_sentence}",
            
    ]
    
    
    
    DESCRIPTOR_TEMPLATE=[
        # "{sentence}",
        "rendered video. {sentence}",
        "synthetic video. {sentence}",
        "this video is rendered using Game Engine. {sentence}",
        "the video is synthetic. {sentence}",
        "this is a synthetic video created with Game Engine. {sentence}",
        "note: this video is synthetically rendered using Game Engine. {sentence}",
        "{sentence}. note: this video is synthetically rendered using Game Engine",
        "generated using Game Engine. {sentence}",
        "{sentence}. generated using Game Engine",
        "it is rendered video and is not a real photograph. {sentence}",
        "the content of this video is rendered. {sentence}",
        "{sentence}. the content of this video is rendered",
        "this video is a virtual render produced. {sentence}",
        "{sentence}. this video is a virtual render produced",
        # More templates if needed...
    ]

    
    lemmatizer = WordNetLemmatizer()
    
    @classmethod
    def get_action_description(cls,scene_type,action_type):
        if scene_type=="ground":
            if action_type=="move":
                action_word_list=cls.GROUND_MOVE_WORD_LIST
                
            elif action_type=="jump":
                action_word_list=cls.JUMP_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.GROUND_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
        
        elif scene_type=="near_ground":
            if action_type=="near_ground_fly":
                action_word_list=cls.NEAR_GROUND_FLY_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.NEAR_GROUND_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
        
        elif scene_type=="sky":
            if action_type=="fly":
                action_word_list=cls.FLY_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.SKY_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        elif scene_type=="overwater":
            if action_type=="move":
                action_word_list=cls.OVERWATER_MOVE_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.OVERWATER_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        elif scene_type=="underwater":
            if action_type=="swim":
                action_word_list=cls.SWIM_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.UNDERWATER_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        else:
            raise NotImplementedError("not implement")            

        
        action_description=random.choice(action_word_list)
        return action_description
                
    @classmethod
    def assemble_description(cls,scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list):
        
        # template=random.choice(UnrealTrajLoraDataset.TEMPLATE)
        
        background_description=background_description.lower()
        
        obj_num=len(object_description_list)
        assert len(action_description_list)==obj_num and len(camera_pose_description_list)==obj_num and len(action_type_list)==obj_num
        
        
        obj_sentence_list,cam_sentence_list=[],[]
        for object_description,action_description,action_type,camera_pose_description in zip(object_description_list,action_description_list,action_type_list,camera_pose_description_list):
            object_description=object_description.lower()
            
            for word in ["a ","the "," a "," the "]:
                if word in object_description:
                    if object_description.startswith(word):
                        object_description.replace(word,"")
        
            action_type=action_type.lower()
            action_description=action_description.lower()
            
            if not action_description:
                action_description=cls.get_action_description(scene_type,action_type)

            assert action_description!=""
            
            
            use_adj=random.choice([True,False])
            if use_adj:
                obj_template_list=UnrealTrajLoraDataset.ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajLoraDataset.ADJ_CAM_TEMPLATE
            else:
                obj_template_list=UnrealTrajLoraDataset.NO_ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajLoraDataset.NO_ADJ_CAM_TEMPLATE
            
            obj_template,cam_template=random.choice(obj_template_list),random.choice(cam_template_list)
            
            
            a_the_str=random.choice(UnrealTrajLoraDataset.A_THE_LIST)
            
            
            verb = action_description
            base_verb = cls.lemmatizer.lemmatize(verb, pos='v')
            
            action_name_present_str=get_third_person_singular(base_verb)
            # stemmer = PorterStemmer()
            # action_name_ing_str = stemmer.stem(base_verb) + "ing"
            action_name_ing_str=get_present_continuous(base_verb)
            
            

            if "action_name" in obj_template:
                if "action_name_ing" in obj_template:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
                else:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name=action_name_present_str)
            else:
                obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description)

            
            front,left,top=camera_pose_description.split("_")
            
            front_left_top_list=[]
            view_point_sentence=""
            for s in [front,left,top]:
                if s:
                    front_left_top_list.append(s)
            
            view_str_num=random.randint(1,len(front_left_top_list))
            # view_str_num=random.randint(len(front_left_top_list),len(front_left_top_list))
            
            chosen_front_left_top_list=random.sample(front_left_top_list,k=view_str_num)
            
            view_point_sentence=" ".join(chosen_front_left_top_list)
            
            cam_sentence=cam_template.format(object_sentence=obj_sentence,view_point_sentence=view_point_sentence)
            
            
            cam_sentence=[i for i in cam_sentence.split(" ") if len(i)>0]
            cam_sentence=" ".join(cam_sentence)
            cam_sentence_list.append(cam_sentence)
            
            obj_sentence=[i for i in obj_sentence.split(" ") if len(i)>0]
            obj_sentence=" ".join(obj_sentence)
            obj_sentence_list.append(obj_sentence)
        
        
        
        use_back=random.choice([True,False])
        
        if obj_num==1:
            no_cam_des=random.choice([True,False])
            #no_cam_des=True
            if no_cam_des:
                sentence_list=obj_sentence_list
            else:
                sentence_list=cam_sentence_list
            if use_back:
                ass_template_list=UnrealTrajLoraDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
            else:
                ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(first_sentence=sentence_list[0])
            
        else:
            if use_back:
                ass_template_list=UnrealTrajLoraDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                no_cam_des=random.choice([True,False])
                #no_cam_des=True
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                for sentence_idx in range(1,len(cam_sentence_list)):
                    no_cam_des=random.choice([True,False])
                    #no_cam_des=True
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajLoraDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                    
            else:
                ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                ass_template=random.choice(ass_template_list)
                no_cam_des=random.choice([True,False])
                #no_cam_des=True
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                    
                description=sentence_list[0]
                for sentence_idx in range(1,len(cam_sentence_list)):
                    no_cam_des=random.choice([True,False])
                    #no_cam_des=True
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    
                    ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajLoraDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
        
        return description 
    
    
    @classmethod
    def assemble_description_without_cam(cls,scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list):
        

        
        background_description=background_description.lower()
        
        obj_num=len(object_description_list)
        assert len(action_description_list)==obj_num and len(camera_pose_description_list)==obj_num and len(action_type_list)==obj_num
        
        
        obj_sentence_list,cam_sentence_list=[],[]
        for object_description,action_description,action_type,camera_pose_description in zip(object_description_list,action_description_list,action_type_list,camera_pose_description_list):
            object_description=object_description.lower()
            
            for word in ["a ","the "," a "," the "]:
                if word in object_description:
                    if object_description.startswith(word):
                        object_description.replace(word,"")
        
            action_type=action_type.lower()
            action_description=action_description.lower()
            
            if not action_description:
                action_description=cls.get_action_description(scene_type,action_type)

            assert action_description!=""
            
            
            use_adj=random.choice([True,False])
            if use_adj:
                obj_template_list=UnrealTrajLoraDataset.ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajLoraDataset.ADJ_CAM_TEMPLATE
            else:
                obj_template_list=UnrealTrajLoraDataset.NO_ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajLoraDataset.NO_ADJ_CAM_TEMPLATE
            
            obj_template,cam_template=random.choice(obj_template_list),random.choice(cam_template_list)
            
            
            a_the_str=random.choice(UnrealTrajLoraDataset.A_THE_LIST)
            
            
            verb = action_description
            base_verb = cls.lemmatizer.lemmatize(verb, pos='v')
            
            action_name_present_str=get_third_person_singular(base_verb)
            # stemmer = PorterStemmer()
            # action_name_ing_str = stemmer.stem(base_verb) + "ing"
            action_name_ing_str=get_present_continuous(base_verb)
            
            

            if "action_name" in obj_template:
                if "action_name_ing" in obj_template:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
                else:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name=action_name_present_str)
            else:
                obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description)

            
            front,left,top=camera_pose_description.split("_")
            
            front_left_top_list=[]
            view_point_sentence=""
            for s in [front,left,top]:
                if s:
                    front_left_top_list.append(s)
            
            view_str_num=random.randint(1,len(front_left_top_list))
            # view_str_num=random.randint(len(front_left_top_list),len(front_left_top_list))
            
            chosen_front_left_top_list=random.sample(front_left_top_list,k=view_str_num)
            
            view_point_sentence=" ".join(chosen_front_left_top_list)
            
            cam_sentence=cam_template.format(object_sentence=obj_sentence,view_point_sentence=view_point_sentence)
            
            
            cam_sentence=[i for i in cam_sentence.split(" ") if len(i)>0]
            cam_sentence=" ".join(cam_sentence)
            cam_sentence_list.append(cam_sentence)
            
            obj_sentence=[i for i in obj_sentence.split(" ") if len(i)>0]
            obj_sentence=" ".join(obj_sentence)
            obj_sentence_list.append(obj_sentence)
        
        
        
        use_back=True
        
        no_cam_des=True
        if obj_num==1:
            if no_cam_des:
                sentence_list=obj_sentence_list
            else:
                sentence_list=cam_sentence_list
            if use_back:
                ass_template_list=UnrealTrajLoraDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
            else:
                ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(first_sentence=sentence_list[0])
            
        else:
            if use_back:
                ass_template_list=UnrealTrajLoraDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                for sentence_idx in range(1,len(cam_sentence_list)):
                    
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajLoraDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                    
            else:
                ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                ass_template=random.choice(ass_template_list)
                
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                    
                description=sentence_list[0]
                for sentence_idx in range(1,len(cam_sentence_list)):
                    
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    
                    ass_template_list=UnrealTrajLoraDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajLoraDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
        
        return description 
    
    SCENE_TYPE_DES_MAP={
        "sky":[
        "sky",
        "blue sky",
        "fluffy clouds",
        "golden sunset",
        "starry night",
        "vibrant sunrise",
        "soft twilight",
        "cumulus clouds",
        "overcast sky",
        "dark storm clouds",
        "pale dawn",
        "radiant twilight",
        "painted sky",
        ],
        
        "ground":[
        "ground",
        "grass",
        "meadow",
        "sunny meadow",
        "forest trail",
        "forest",
        "beach",
        "sandy beach",
        "desert oasis",
        "desert",
        "snowfield",
        "snow",
        "urban park",
        "street",
        "urban street",
        "road",
        "country road",
        "garden",
        "flower garden",
        "playground",
        "gym",
        "amusement park",
        ],
        
        "near_ground":[
        "ground",
        "grass",
        "meadow",
        "sunny meadow",
        "forest trail",
        "forest",
        "beach",
        "sandy beach",
        "desert oasis",
        "desert",
        "snowfield",
        "snow",
        "urban park",
        "street",
        "urban street",
        "road",
        "country road",
        "garden",
        "flower garden",
        "playground",
        "gym",
        "amusement park",
        ],
        
        "overwater":[
        "overwater",
        "over water",
        "water surface",
        "sea surface",
        "pool surface",
        "ocean surface",
        "pool"
        "sea",
        "ocean",
        ],
        
        "underwater":[
        "underwater",
        "under water",
        "coral reef",
        "ocean floor",
        "seagrass bed",
        "underwater cave",
        "marine trench",
        "deep sea",
        "sandy seabed",
        "kelp forest",
        ]
    }
    
    SCENE_TYPE_OBJ_DES_MAP={
        "sky":{
            "fly":[
                "rocket","airplane","glider","fighterjet","Missile",
                "Aerocraft","UAV",
                "Bat","bird","parrot","eagle",
                "angrybird",
            ],
            "idle":["balloon","Airship","hot air balloon","cloud",],
            "fly;idle":[
                "ironman",
                "UFO","aircraft","helicopter",
                "dragon","pterosaur","phoenix","Pegasus","angel",
            ]
        
                        
        },
        "near_ground":{
            "near_ground_fly;idle":["drone","balloon",
                                    "Moths","fly","mosquito","bee","butterfly",
                                    "ghost","fairy",
                                    "little bird",
                                ]
        },
        "ground":{
            "idle":["Shampoo","Trophy","hourglass","chessboard","calculator","spraypaint","Turntable","candle","Stapler","camera","telephone","hamburger","flashlight","can","scissor","book","notebook","compass","pokeball","cup","watch","mug","egg","toy","hat","alarmclock","hotdog","plate","bottle",
                    "Nightstand","basket","Stool","toaster","birdcage","antenna","microwave","Cage","Well","TreasureChest","chest","barrel","trashcan","typewriter","Microscope","drawer","helmet","bomb","balloon","lamp","Campfire","stone","laptop","vase","Gramophone","fruit","bowl","lantern","suitcase","ball","chest","box","cube","sphere","cylinder","fan","mirror","plane","TV","Television",
                    "TelephoneBooth","Cauldron","closestool","bookshelf","PoolTable","fireplace","lawnmower","robotic","cabinet","VendingMachine","Billboard","bench","table","chair","desk","printer","gate","door","fridge","washingMachine","machine","clock",
                    "Trebuchet","LampPost","Satellite","device","bed","Bell","Turret",
                    "station","Turbine","lighthouse","house","fort","Gazebo", "pyramid","building","windmill","waterwheel","waterwill","Ferriswheel","Ferris","playground","statue",
                    "sunflower","plant","pumpkin","flower","grass","fire","beet","corn","potato","mushroom","Tomato",
                    "tree","bamboo",],
            "idle;move;jump":[
                    "dinosaur","moose","wolf","deer","horse","leopard","antelope","lion","tiger",
                    "hamster","mouse","rat","Squirrel",
                    "Raccoon","zebra","sheep","beast","frog","toad","Kangaroo","kong","Bulldog","elephant","chameleon","bear","panda","dog","badger","cat","mole","fox","monkey","rabbit","bunny","chicken","Chimpanzee","Orangutan","Gorilla",
                    "character","human","person",
                    "man","woman","male","female","boy","girl",
                    "Nymph","naruto","wolfman","chimera","monster","robot","ogre","skeleton","alien","zombie","shrek","santa",
                    "elder","Granny","baby",
                    "boxer","maid","guard","Wrestler","Magician","Scientist","pirate","Clown","Firefighter","cook","Pharaoh","cowboy","trollWarrior","villager","actor","swat","chef","captain","hero","mage","ninja","goalkeeper","viking","astronaut","worker","nurse","nun","farmer","doctor","warrior","butcher","knight","witch","wizard","pilot","racer","athlete","sportsman","police","policeman","driver","soldier",
                    "king","princess",
                    "groot","deadpool","spider-man","batman","ironman",
                    ],
            "idle;move":["snail","worm","spider","ant","Scorpion","locust",
                        "sloth","lizard",
                        "giraffe","Hippopotamus","Rhinoceros","Rhino","cow","donkey","llama","pig","Mammoth",
                        "truck","tank","car","van",]
        },
        "overwater":{
            "idle":[
                "lotus",
            ],
            "move;idle":[
                "ship","boat","dolphin","seal","whale","duck",
                "crab","Lobster","turtle",
            ]
            
        },
        "underwater":{
            "swim;idle":[
                "submarine",
                "dolphin","Anglerfish","fish","jellyfish","shark","penguin",
                "seal","whale",
                "crab","Lobster","octopus","turtle",
            ]
        }
    }
    
    
    REAL_SCENE_TYPE_OBJ_DES_MAP={
        "sky":{
            "fly":[
                "eagle",
                "Boeing",
            ],
            "idle":["hot air balloon"],                        
        },
        "near_ground":{
            "near_ground_fly;idle":["bee",
                                ]
        },
        "ground":{
            "idle":["tree","house","building","desk","table"],
            "idle;move;jump":[
                    "dog","cat","man","man with hat","girl","monkey"
                    ],
            "idle;move":["pig","car","truck","red car","blue car"]
        },
        "overwater":{

            "move;idle":[
                "ship","boat","dolphin"
            ]
            
        },
        "underwater":{
            "swim;idle":[
                "submarine",
                "dolphin","Anglerfish","fish","jellyfish","shark","octopus","turtle",
            ]
        }
    }
    

    
    @classmethod
    def create_validation_prompts(cls,num,use_synthetic_des,max_obj_num=3):
        
        def _gen_prompt():
            scene_type=random.choice(list(cls.SCENE_TYPE_DES_MAP.keys()))
            background_description=random.choice(cls.SCENE_TYPE_DES_MAP[scene_type])
            
            object_description_list=[]
            action_description_list=[]
            action_type_list=[]
            camera_pose_description_list=[]
            
            obj_num=random.randint(1,max_obj_num)
            
            for i in range(obj_num):
                ord_action_type_list=random.choice(list(cls.SCENE_TYPE_OBJ_DES_MAP[scene_type].keys()))
                _action_type_list=ord_action_type_list.split(";")
                action_type=random.choice(_action_type_list)
                action_type_list.append(action_type)
                
                action_description=cls.get_action_description(scene_type,action_type)
                action_description_list.append(action_description)
                
                obj_description=random.choice(cls.SCENE_TYPE_OBJ_DES_MAP[scene_type][ord_action_type_list])
                object_description_list.append(obj_description)
                

                while True:
                    top_str_list=["top",""]
                    side_str_list=["left","right",""]
                    front_str_list=["back","front",""]
                    
                    top_str=random.choice(top_str_list)
                    side_str=random.choice(side_str_list)
                    front_str=random.choice(front_str_list)
                    
                    if not all([s=="" for s in [top_str,side_str,front_str]]):
                        break
                camera_pose_description_list.append("_".join([top_str,side_str,front_str]))
                                
        
            description=cls.assemble_description(scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list)
            
            if use_synthetic_des:
                descriptor_template=random.choice(UnrealTrajLoraDataset.DESCRIPTOR_TEMPLATE)
                description=descriptor_template.format(sentence=description)
            return description
        
        prompts=[]
            
        for i in range(num):
            
            prompts.append(_gen_prompt())    
        
        return prompts

    
    @classmethod
    def create_validation_prompts_without_cam(cls,num,use_synthetic_des,min_obj_num=1,max_obj_num=3):
        
        def _gen_prompt():
            scene_type=random.choice(list(cls.SCENE_TYPE_DES_MAP.keys()))
            # scene_type="underwater"
            background_description=random.choice(cls.SCENE_TYPE_DES_MAP[scene_type])
            
            object_description_list=[]
            action_description_list=[]
            action_type_list=[]
            camera_pose_description_list=[]
            
            obj_num=random.randint(min_obj_num,max_obj_num)
            
            for i in range(obj_num):
                ord_action_type_list=random.choice(list(cls.SCENE_TYPE_OBJ_DES_MAP[scene_type].keys()))
                _action_type_list=ord_action_type_list.split(";")
                action_type=random.choice(_action_type_list)
                action_type_list.append(action_type)
                
                action_description=cls.get_action_description(scene_type,action_type)
                action_description_list.append(action_description)
                
                obj_description=random.choice(cls.SCENE_TYPE_OBJ_DES_MAP[scene_type][ord_action_type_list])
                object_description_list.append(obj_description)
                

                while True:
                    top_str_list=["top",""]
                    side_str_list=["left","right",""]
                    front_str_list=["back","front",""]
                    
                    top_str=random.choice(top_str_list)
                    side_str=random.choice(side_str_list)
                    front_str=random.choice(front_str_list)
                    
                    if not all([s=="" for s in [top_str,side_str,front_str]]):
                        break
                camera_pose_description_list.append("_".join([top_str,side_str,front_str]))
                                
        
            description=cls.assemble_description_without_cam(scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list)
            
            if use_synthetic_des:
                descriptor_template=random.choice(UnrealTrajLoraDataset.DESCRIPTOR_TEMPLATE)
                description=descriptor_template.format(sentence=description)
            return description
        
        prompts=[]
            
        for i in range(num):
            
            prompts.append(_gen_prompt())    
        
        return prompts
        
    
    
    
    def __init__(
        self,
        # root_path,
        data_root,
        lable_root,
        mask_root,
        # json_path,
        seq_csv_root,
        hdri_json_file_path,
        asset_json_file_path,
        single_static_num=6000,
        single_dynamic_num=8000,
        multi_static_num=6000,
        multi_dynamic_num=6000,
        sample_stride=4,
        sample_n_frames=16,
        sample_size=[256, 384],
        is_image=True,
        use_flip=True,
    ):
        # self.root_path = root_path
        self.data_root=data_root
        self.lable_root=lable_root
        self.mask_root=mask_root

        
        self.seq_csv_root=seq_csv_root
        self.hdri_json_file_path=hdri_json_file_path
        
        with open(hdri_json_file_path,"r") as f:
            self.hdri_json_data=json.load(f)
        
        with open(asset_json_file_path,"r") as f:
            self.asset_json_data=json.load(f)
        self.asset_json_file_path=asset_json_file_path
        
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        self.single_static_num=single_static_num
        self.single_dynamic_num=single_dynamic_num
        self.multi_static_num=multi_static_num
        self.multi_dynamic_num=multi_dynamic_num
        self.length=0
        self.dataset=[]
        self.data_type_list=[]
        self.seq_id_list=[]
        

        for single_type in ["single","multi"]:
            for static_type in ["static","dynamic"]: ##format of key is {single/multi}_{static/dynamic}


                
                if single_type=="single" and static_type=="static":
                    num=self.single_static_num
                    static_type="static"
                    multi_suffix=""
                
                if single_type=="single" and static_type=="dynamic":
                    num=self.single_dynamic_num
                    static_type="dynamic"
                    multi_suffix=""
                
                if single_type=="multi" and static_type=="static":
                    num=self.multi_static_num
                    static_type="static"
                    multi_suffix="_multi"
                    
                if single_type=="multi" and static_type=="dynamic":
                    num=self.multi_dynamic_num
                    static_type="dynamic"
                    multi_suffix="_multi"
                    
                for i in range(num):
                    annotation_file_path=os.path.join(self.lable_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{i}.json")
                    clip_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{i}")
                    self.dataset.append({
                        "annotation_file_path":annotation_file_path,
                        "clip_path":clip_path,
                    })
                    self.data_type_list.append("_".join([single_type,static_type]))
                    self.seq_id_list.append(str(i))
                
        self.length=len(self.dataset)
            
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            
            mask_transforms = [transforms.Resize(sample_size),
                    transforms.RandomHorizontalFlip(),
                    ]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            mask_transforms = [transforms.Resize(sample_size)
            ]
            

        self.pixel_transforms = transforms.Compose(pixel_transforms)
        
        self.mask_transforms = transforms.Compose(mask_transforms)

        
        
        self.seq_meta_data_map=self._get_csv_meta_data_map()
    
    
    def _get_csv_meta_data_map(self):
        
        seq_meta_data_map={}
        for dynamic_type_str in ["static","dynamic"]:
            for single_type_str in ["","_multi"]:
                seq_meta_data={}
                csv_path=os.path.join(self.seq_csv_root,f"traj_{dynamic_type_str}{single_type_str}.csv")
                with open(csv_path, mode="r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    csv_rows = list(
                        csv_reader
                    )

                for row_index, row in enumerate(csv_rows):
                    if row["Type"] == "Group":

                        seq_id = row["Seq_ID"]
                        body_id=-1
                        row.pop("Seq_ID")
                        seq_meta_data[seq_id]={}
                        seq_meta_data[seq_id]["camera"]=row
                        seq_meta_data[seq_id]["objects"]={}
                    else:

                        body_id+=1
                        row.pop("Seq_ID")
                        seq_meta_data[seq_id]["objects"][str(body_id)]=row

                        
                
                if dynamic_type_str=="static":
                    dynamic_type="static"
                else:
                    dynamic_type="dynamic"
                    
                if single_type_str=="_multi":
                    single_type="multi"
                else:
                    single_type="single"
                key_name=f"{single_type}_{dynamic_type}"
                
                seq_meta_data_map[key_name]=seq_meta_data
            

        return seq_meta_data_map

    def load_img_list(self, idx):
        video_dict = self.dataset[idx]
        # video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_path=video_dict['clip_path']
        frame_files = sorted(
            [
                os.path.join(video_path, f)
                for f in os.listdir(video_path)
                if os.path.isfile(os.path.join(video_path, f)) and f.endswith(".png") and "-" not in f
            ]
        )

        

        return frame_files

    def get_text_prompt_and_mask_list(self,idx,time_idx):
        data_type=self.data_type_list[idx]
        label_data=self.dataset[idx]
        seq_id=self.seq_id_list[idx]
        annotation_file_path=label_data["annotation_file_path"]
        with open(annotation_file_path,"r") as f:
            annotation_data=json.load(f)
        

        seq_meta_data=self.seq_meta_data_map[data_type][seq_id]
        
        background_description=get_background_description(self.hdri_json_data,seq_meta_data["camera"])
        if "static" in data_type:
            static_type="static"
        else:
            static_type="dynamic"
            
        if "multi" in data_type:
            multi_suffix="_multi"
        else:
            multi_suffix=""
        
        data_type=self.data_type_list[idx]

        seq_id=self.seq_id_list[idx]
        _kwargs={
            "data_type":data_type,
            "seq_id":seq_id,
            "max_num":None,
        }
            
        # exr_file_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,"exr",f"{time_idx:04}.exr") ##traj_dataset/Rendered_Traj_Results
        # seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description(exr_file_path,self.asset_json_data,seq_meta_data,time_idx,**kwargs)
        
        mask_root=os.path.join(self.mask_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,str(time_idx))
        seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v3(mask_root,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**_kwargs)

        
        
        # mask_root="temp_mask"

        # total_img = Image.fromarray(total_mask[...,0].astype(np.uint8)*255)
        # save_dir=os.path.join(mask_root,data_type,seq_id)
        # os.makedirs(save_dir,exist_ok=True)
        # total_img.save(os.path.join(save_dir,f"{time_idx}-total.png"))
        
        # for mask_idx,mask in enumerate(obj_mask_list):
            
        #     mask=mask[...,0]
        #     img = Image.fromarray(mask.astype(np.uint8)*255)

        #     img.save(os.path.join(save_dir,f"{time_idx}-{mask_idx}.png"))
        
        
        comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
        scene_type=comment_dict["scene_type"]
    
        if len(seen_obj_idx_list)>0:
            camera_pose_description_list=get_camera_pose_description_v2(annotation_data,seen_obj_idx_list,time_idx)
            # total_description=self.assemble_description(scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list)
            total_description=self.assemble_description_without_cam(scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list)
            use_descriptor_num=random.uniform(0,1)
            if use_descriptor_num<0.9:
                use_descriptor=True
            else:
                use_descriptor=False
            
            if use_descriptor:
                descriptor_template=random.choice(UnrealTrajLoraDataset.DESCRIPTOR_TEMPLATE)
                total_description=descriptor_template.format(sentence=total_description)
        else:
            total_description=""
            

        
        return total_description,total_mask,obj_mask_list


    def get_batch(self, idx):
        img_path_list = self.load_img_list(idx)[:-1]
        total_frames = len(img_path_list)


        if self.is_image:
            frame_indice = random.randint(0, total_frames - 1)
        else:
            if isinstance(self.sample_stride, int):
                current_sample_stride = self.sample_stride
            else:
                assert len(self.sample_stride) == 2
                assert (self.sample_stride[0] >= 1) and (self.sample_stride[1] >= self.sample_stride[0])
                current_sample_stride = random.randint(self.sample_stride[0], self.sample_stride[1])

            cropped_length = self.sample_n_frames * current_sample_stride
            start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
            end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        caption,total_mask,obj_mask_list=self.get_text_prompt_and_mask_list(idx,frame_indice)
        
        img_path=img_path_list[frame_indice]
        image=Image.open(img_path_list[frame_indice])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
        pixel_values = pixel_values / 255.
        
        total_mask=torch.from_numpy(total_mask).permute(2, 0, 1).contiguous()
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(torch.from_numpy(obj_mask).permute(2, 0, 1).contiguous())
            
        obj_mask_list=new_obj_mask_list


        return img_path,pixel_values, caption,total_mask,obj_mask_list

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        while True:
            img_path,image, image_caption,total_mask,obj_mask_list = self.get_batch(idx)
            if image_caption!="":
                break
            else:
                idx = random.randint(0, self.length - 1)
                continue
    
        image = self.pixel_transforms(image)
        
        total_mask=self.mask_transforms(total_mask)
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(self.mask_transforms(obj_mask))
        obj_mask_list=new_obj_mask_list
        # sample = dict(img_path=img_path,image=image, caption=image_caption,total_mask=total_mask,obj_mask_list=obj_mask_list)
        sample = dict(img_path=img_path,image=image, caption=image_caption)

        return sample

class UnrealTrajVideoDataset(Dataset):
    
    OBJ_CONCAT_LIST=[
        "{sentence_1} and {sentence_2}",
        "{sentence_1} as well as {sentence_2}",
        "{sentence_1} along with {sentence_2}",
        "{sentence_1} together with {sentence_2}"
    ]
    
    
    ENTER_TEMPLATE=[
        "{objects} appear on the screen",
        "{objects} enter the frame",
        "{objects} come into view",
        "{objects} come into sight",
        "{objects} emerge into the scene",
        "{objects} show up on the display",
        "the screen reveal {objects}",
        "{objects} materialize on screen",
        "{objects} pop into the picture",
    ]
    
    

    EXIT_TEMPLATE=[
        "{objects} disappear from view",
        "{objects} move out of sight",
        "{objects} exit the screen",
        "{objects} go off-screen",
        "{objects} leave the field of vision",
        "{objects} vanish from sight",
        "{objects} are no longer visible",
        "{objects} left the frame",
        "{objects} move beyond the visible area",
        "{objects} go out of view"
    ]
        

    THEN_TEMPLATE=[
        "{sentence_1}. then, {sentence_2}",
        "{sentence_1}. subsequently, {sentence_2}",
        "{sentence_1}. next, {sentence_2}",
        "{sentence_1}. after that, {sentence_2}",
        "{sentence_1}. in following, {sentence_2}",
        "{sentence_1}. later {sentence_2}",
        "{sentence_1}. afterwards, {sentence_2}",
    ]

    ENTER_EXIT_CONCAT_LIST=[
        ". at the same time,",
        ", and",
        ", while",
        ", as",
        ". simultaneously,",
        ". meanwhile,"
        ". in the meantime,"
        ". concurrently,"

    ]
    
    GROUND_MOVE_WORD_LIST=[
        "move",
        
        "walk",
        "shift",
        "stroll",
        
        "run",
        "dash",
        "sprint",
    ]
    
    OVERWATER_MOVE_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "glide",
        "swim",
        # "sail",
    ]
    
    JUMP_WORD_LIST=[
        "jump",
        "leap",
    ]
    
    FLY_WORD_LIST=[
        "move",
        
        "shift",
        
        "fly",
        "soar",
        "glide",
        
        # "drift",
        # "hover",
        # "flap",
        # "swoop",
        # "ascend",
        # "descend",
        # "flutter",
    ]
    
    NEAR_GROUND_FLY_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "fly",        
        "glide",
        
    ]
    
    SWIM_WORD_LIST=[
        "move",
        
        "shift",
        "drift",
        
        "swim",
        "dive",
        
    ]

    
    GROUND_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        # "pause",
        "halt",

    ]
    
    
    SKY_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "hover",
        "suspend",
        
    ]
    
    NEAR_GROUND_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "hover",
        "suspend",
    ]
    
    
    OVERWATER_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "suspend",
    ]
    
    UNDERWATER_IDLE_WORD_LIST=[
        "idle",
        "rest",
        "stay",
        "remain",
        "pause",
        "halt",
        
        "float",
        "suspend",
    ]
    

    A_THE_LIST=["a","the"]
    
    # CONCAT_LIST=["and",", and","while",", while","."]
    CONCAT_LIST=["and",", and","."]
    
    ADJ_OBJ_TEMPLATE=[
        "{a_the} {object_name} which is {action_name_ing}",
        "{a_the} {action_name_ing} {object_name}",
        "{a_the} {object_name}",
    ]
    
    NO_ADJ_OBJ_TEMPLATE=[
        "{a_the} {object_name} {action_name}",
        "{a_the} {object_name} is {action_name_ing}"
    ]
    
    
    
    ADJ_CAM_TEMPLATE=[ ##no present tense
        # "{object_sentence} from {view_point_sentence} view",

        "{object_sentence} is viewed from {view_point_sentence} side",

        "{object_sentence} is observed from the {view_point_sentence} perspective",
        
        "camera captures {object_sentence} as seen from {view_point_sentence} perspective",

        "{object_sentence} is seen from {view_point_sentence} side",

        "{object_sentence} is viewed from {view_point_sentence} viewpoint",

        "{object_sentence} is captured at the {view_point_sentence} angle",

        # "the camera's {view_point_sentence} shot shows the {object_sentence}"
    ]
    
    NO_ADJ_CAM_TEMPLATE=[
        # "{object_sentence} from {view_point_sentence} side",

        "viewed from {view_point_sentence} perspective, {object_sentence}",
        
        "observed from {view_point_sentence} view, {object_sentence}",
        
        "seen from {view_point_sentence} view, {object_sentence}",
    ]

    
    BACK_ASSEMBLE_SINGLE_TEMPLATE=[
        "in {background}, {first_sentence}",
        "with {background} behind, {first_sentence}",
        "against {background}, {first_sentence}",
        "in front of {background}, {first_sentence}",        
        "with {background} in the background, {first_sentence}",
        "with {background} backdrop, {first_sentence}",

        "{first_sentence}, in {background}",
        "{first_sentence}, with {background} behind",
        "{first_sentence}, against {background}",
        "{first_sentence}, in front of {background}",  
        "{first_sentence}, with {background} in the background",
        "{first_sentence}, with {background} backdrop",

        
    ]
    
    NO_BACK_ASSEMBLE_SINGLE_TEMPLATE=[
        "{first_sentence}",    
    ]
    

    
    NO_BACK_ASSEMBLE_MULTI_TEMPLATE=[        
        "{first_sentence} {concat} {second_sentence}",
            
    ]
    
    
    DESCRIPTOR_TEMPLATE=[
        # "{sentence}",
        "rendered video. {sentence}",
        "synthetic video. {sentence}",
        "this video is rendered using Game Engine. {sentence}",
        "the video is synthetic. {sentence}",
        "this is a synthetic video created with Game Engine. {sentence}",
        "note: this video is synthetically rendered using Game Engine. {sentence}",
        "{sentence}. note: this video is synthetically rendered using Game Engine",
        "generated using Game Engine. {sentence}",
        "{sentence}. generated using Game Engine",
        "it is rendered video and is not a real photograph. {sentence}",
        "the content of this video is rendered. {sentence}",
        "{sentence}. the content of this video is rendered",
        "this video is a virtual render produced. {sentence}",
        "{sentence}. this video is a virtual render produced",
        # More templates if needed...
    ]

    
    lemmatizer = WordNetLemmatizer()
    
    @classmethod
    def get_action_description(cls,scene_type,action_type):
        if scene_type=="ground":
            if action_type=="move":
                action_word_list=cls.GROUND_MOVE_WORD_LIST
                
            elif action_type=="jump":
                action_word_list=cls.JUMP_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.GROUND_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
        
        elif scene_type=="near_ground":
            if action_type=="near_ground_fly":
                action_word_list=cls.NEAR_GROUND_FLY_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.NEAR_GROUND_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
        
        elif scene_type=="sky":
            if action_type=="fly":
                action_word_list=cls.FLY_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.SKY_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        elif scene_type=="overwater":
            if action_type=="move":
                action_word_list=cls.OVERWATER_MOVE_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.OVERWATER_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        elif scene_type=="underwater":
            if action_type=="swim":
                action_word_list=cls.SWIM_WORD_LIST
            
            elif action_type=="idle":
                action_word_list=cls.UNDERWATER_IDLE_WORD_LIST
            
            else:
                raise NotImplementedError("not implement") 
            
        else:
            raise NotImplementedError("not implement")            

        
        action_description=random.choice(action_word_list)
        return action_description
    
    
    @classmethod
    def get_seen_objs_description(cls,scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list):
        
        # template=random.choice(UnrealTrajLoraDataset.TEMPLATE)
        
        background_description=background_description.lower()
        
        obj_num=len(object_description_list)
        assert len(action_description_list)==obj_num and len(camera_pose_description_list)==obj_num and len(action_type_list)==obj_num
        
        
        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=[],[],[],[]
        for object_description,action_description,action_type,camera_pose_description in zip(object_description_list,action_description_list,action_type_list,camera_pose_description_list):
            object_description=object_description.lower()
            
            for word in ["a ","the "," a "," the "]:
                if word in object_description:
                    if object_description.startswith(word):
                        object_description.replace(word,"")
        
            action_type=action_type.lower()
            action_description=action_description.lower()
            
            if not action_description:
                action_description=cls.get_action_description(scene_type,action_type)

            assert action_description!=""
            
            
            use_adj=random.choice([True,False])
            if use_adj:
                obj_template_list=UnrealTrajVideoDataset.ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajVideoDataset.ADJ_CAM_TEMPLATE
            else:
                obj_template_list=UnrealTrajVideoDataset.NO_ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajVideoDataset.NO_ADJ_CAM_TEMPLATE
            
            obj_template,cam_template=random.choice(obj_template_list),random.choice(cam_template_list)
            
            obj_adj_template=random.choice(UnrealTrajVideoDataset.ADJ_OBJ_TEMPLATE)
            obj_no_adj_template=random.choice(UnrealTrajVideoDataset.NO_ADJ_OBJ_TEMPLATE)
            
            
            a_the_str=random.choice(UnrealTrajVideoDataset.A_THE_LIST)
            
            
            verb = action_description
            base_verb = cls.lemmatizer.lemmatize(verb, pos='v')
            
            action_name_present_str=get_third_person_singular(base_verb)
            # stemmer = PorterStemmer()
            # action_name_ing_str = stemmer.stem(base_verb) + "ing"
            action_name_ing_str=get_present_continuous(base_verb)
            
            

            if "action_name" in obj_template:
                if "action_name_ing" in obj_template:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
                else:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name=action_name_present_str)
            else:
                obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description)

            
            obj_adj_sentence=obj_adj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
            if "action_name_ing" in obj_no_adj_template:
                obj_no_adj_sentence=obj_no_adj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
            else:
                obj_no_adj_sentence=obj_no_adj_template.format(a_the=a_the_str,object_name=object_description,action_name=action_name_present_str)

            
            front,left,top=camera_pose_description.split("_")
            
            front_left_top_list=[]
            view_point_sentence=""
            for s in [front,left,top]:
                if s:
                    front_left_top_list.append(s)
            
            view_str_num=random.randint(1,len(front_left_top_list))
            # view_str_num=random.randint(len(front_left_top_list),len(front_left_top_list))
            
            chosen_front_left_top_list=random.sample(front_left_top_list,k=view_str_num)
            
            view_point_sentence=" ".join(chosen_front_left_top_list)
            
            cam_sentence=cam_template.format(object_sentence=obj_sentence,view_point_sentence=view_point_sentence)
            
            
            cam_sentence=[i for i in cam_sentence.split(" ") if len(i)>0]
            cam_sentence=" ".join(cam_sentence)
            cam_sentence_list.append(cam_sentence)
            
            obj_sentence=[i for i in obj_sentence.split(" ") if len(i)>0]
            obj_sentence=" ".join(obj_sentence)
            obj_sentence_list.append(obj_sentence)
            
            obj_adj_sentence=[i for i in obj_adj_sentence.split(" ") if len(i)>0]
            obj_adj_sentence=" ".join(obj_adj_sentence)
            obj_adj_sentence_list.append(obj_adj_sentence)
            
            obj_no_adj_sentence=[i for i in obj_no_adj_sentence.split(" ") if len(i)>0]
            obj_no_adj_sentence=" ".join(obj_no_adj_sentence)
            obj_no_adj_sentence_list.append(obj_no_adj_sentence)
        
        
        return obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list
        
    @classmethod
    def assemble_description(cls,scene_type,background_description,object_description_list,action_description_list,action_type_list,camera_pose_description_list):
        
        # template=random.choice(UnrealTrajLoraDataset.TEMPLATE)
        
        background_description=background_description.lower()
        
        obj_num=len(object_description_list)
        assert len(action_description_list)==obj_num and len(camera_pose_description_list)==obj_num and len(action_type_list)==obj_num
        
        
        obj_sentence_list,cam_sentence_list=[],[]
        for object_description,action_description,action_type,camera_pose_description in zip(object_description_list,action_description_list,action_type_list,camera_pose_description_list):
            object_description=object_description.lower()
            
            for word in ["a ","the "," a "," the "]:
                if word in object_description:
                    if object_description.startswith(word):
                        object_description.replace(word,"")
        
            action_type=action_type.lower()
            action_description=action_description.lower()
            
            if not action_description:
                action_description=cls.get_action_description(scene_type,action_type)

            assert action_description!=""
            
            
            use_adj=random.choice([True,False])
            if use_adj:
                obj_template_list=UnrealTrajVideoDataset.ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajVideoDataset.ADJ_CAM_TEMPLATE
            else:
                obj_template_list=UnrealTrajVideoDataset.NO_ADJ_OBJ_TEMPLATE
                cam_template_list=UnrealTrajVideoDataset.NO_ADJ_CAM_TEMPLATE
            
            obj_template,cam_template=random.choice(obj_template_list),random.choice(cam_template_list)
            
            
            a_the_str=random.choice(UnrealTrajVideoDataset.A_THE_LIST)
            
            
            verb = action_description
            base_verb = cls.lemmatizer.lemmatize(verb, pos='v')
            
            action_name_present_str=get_third_person_singular(base_verb)
            # stemmer = PorterStemmer()
            # action_name_ing_str = stemmer.stem(base_verb) + "ing"
            action_name_ing_str=get_present_continuous(base_verb)
            
            

            if "action_name" in obj_template:
                if "action_name_ing" in obj_template:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name_ing=action_name_ing_str)
                else:
                    obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description,action_name=action_name_present_str)
            else:
                obj_sentence=obj_template.format(a_the=a_the_str,object_name=object_description)

            
            front,left,top=camera_pose_description.split("_")
            
            front_left_top_list=[]
            view_point_sentence=""
            for s in [front,left,top]:
                if s:
                    front_left_top_list.append(s)
            
            view_str_num=random.randint(1,len(front_left_top_list))
            # view_str_num=random.randint(len(front_left_top_list),len(front_left_top_list))
            
            chosen_front_left_top_list=random.sample(front_left_top_list,k=view_str_num)
            
            view_point_sentence=" ".join(chosen_front_left_top_list)
            
            cam_sentence=cam_template.format(object_sentence=obj_sentence,view_point_sentence=view_point_sentence)
            
            
            cam_sentence=[i for i in cam_sentence.split(" ") if len(i)>0]
            cam_sentence=" ".join(cam_sentence)
            cam_sentence_list.append(cam_sentence)
            
            obj_sentence=[i for i in obj_sentence.split(" ") if len(i)>0]
            obj_sentence=" ".join(obj_sentence)
            obj_sentence_list.append(obj_sentence)
        
        
        
        use_back=random.choice([True,False])
        
        if obj_num==1:
            no_cam_des=random.choice([True,False])
            if no_cam_des:
                sentence_list=obj_sentence_list
            else:
                sentence_list=cam_sentence_list
            if use_back:
                ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
            else:
                ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                description=ass_template.format(first_sentence=sentence_list[0])
            
        else:
            if use_back:
                ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                ass_template=random.choice(ass_template_list)
                no_cam_des=random.choice([True,False])
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                for sentence_idx in range(1,len(cam_sentence_list)):
                    no_cam_des=random.choice([True,False])
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                    
            else:
                ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                ass_template=random.choice(ass_template_list)
                no_cam_des=random.choice([True,False])
                if no_cam_des:
                    sentence_list=obj_sentence_list
                else:
                    sentence_list=cam_sentence_list
                    
                description=sentence_list[0]
                for sentence_idx in range(1,len(cam_sentence_list)):
                    no_cam_des=random.choice([True,False])
                    if no_cam_des:
                        sentence_list=obj_sentence_list
                    else:
                        sentence_list=cam_sentence_list
                    
                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
        
        return description 
    
    
    SCENE_TYPE_DES_MAP={
        "sky":[
        "sky",
        "blue sky",
        "fluffy clouds",
        "golden sunset",
        "starry night",
        "vibrant sunrise",
        "soft twilight",
        "cumulus clouds",
        "overcast sky",
        "dark storm clouds",
        "pale dawn",
        "radiant twilight",
        "painted sky",
        ],
        
        "ground":[
        "ground",
        "grass",
        "meadow",
        "sunny meadow",
        "forest trail",
        "forest",
        "beach",
        "sandy beach",
        "desert oasis",
        "desert",
        "snowfield",
        "snow",
        "urban park",
        "street",
        "urban street",
        "road",
        "country road",
        "garden",
        "flower garden",
        "playground",
        "gym",
        "amusement park",
        ],
        
        "near_ground":[
        "ground",
        "grass",
        "meadow",
        "sunny meadow",
        "forest trail",
        "forest",
        "beach",
        "sandy beach",
        "desert oasis",
        "desert",
        "snowfield",
        "snow",
        "urban park",
        "street",
        "urban street",
        "road",
        "country road",
        "garden",
        "flower garden",
        "playground",
        "gym",
        "amusement park",
        ],
        
        "overwater":[
        "overwater",
        "over water",
        "water surface",
        "sea surface",
        "pool surface",
        "ocean surface",
        "pool"
        "sea",
        "ocean",
        ],
        
        "underwater":[
        "underwater",
        "under water",
        "coral reef",
        "ocean floor",
        "seagrass bed",
        "underwater cave",
        "marine trench",
        "deep sea",
        "sandy seabed",
        "kelp forest",
        ]
    }
    
    SCENE_TYPE_OBJ_DES_MAP={
        "sky":{
            "fly":[
                "rocket","airplane","glider","fighterjet","Missile",
                "Aerocraft","UAV",
                "Bat","bird","parrot","eagle",
                "angrybird",
            ],
            "idle":["balloon","Airship","hot air balloon","cloud",],
            "fly;idle":[
                "ironman",
                "UFO","aircraft","helicopter",
                "dragon","pterosaur","phoenix","Pegasus","angel",
            ]
        
                        
        },
        "near_ground":{
            "near_ground_fly;idle":["drone","balloon",
                                    "Moths","fly","mosquito","bee","butterfly",
                                    "ghost","fairy",
                                    "little bird",
                                ]
        },
        "ground":{
            "idle":["Shampoo","Trophy","hourglass","chessboard","calculator","spraypaint","Turntable","candle","Stapler","camera","telephone","hamburger","flashlight","can","scissor","book","notebook","compass","pokeball","cup","watch","mug","egg","toy","hat","alarmclock","hotdog","plate","bottle",
                    "Nightstand","basket","Stool","toaster","birdcage","antenna","microwave","Cage","Well","TreasureChest","chest","barrel","trashcan","typewriter","Microscope","drawer","helmet","bomb","balloon","lamp","Campfire","stone","laptop","vase","Gramophone","fruit","bowl","lantern","suitcase","ball","chest","box","cube","sphere","cylinder","fan","mirror","plane","TV","Television",
                    "TelephoneBooth","Cauldron","closestool","bookshelf","PoolTable","fireplace","lawnmower","robotic","cabinet","VendingMachine","Billboard","bench","table","chair","desk","printer","gate","door","fridge","washingMachine","machine","clock",
                    "Trebuchet","LampPost","Satellite","device","bed","Bell","Turret",
                    "station","Turbine","lighthouse","house","fort","Gazebo", "pyramid","building","windmill","waterwheel","waterwill","Ferriswheel","Ferris","playground","statue",
                    "sunflower","plant","pumpkin","flower","grass","fire","beet","corn","potato","mushroom","Tomato",
                    "tree","bamboo",],
            "idle;move;jump":[
                    "dinosaur","moose","wolf","deer","horse","leopard","antelope","lion","tiger",
                    "hamster","mouse","rat","Squirrel",
                    "Raccoon","zebra","sheep","beast","frog","toad","Kangaroo","kong","Bulldog","elephant","chameleon","bear","panda","dog","badger","cat","mole","fox","monkey","rabbit","bunny","chicken","Chimpanzee","Orangutan","Gorilla",
                    "character","human","person",
                    "man","woman","male","female","boy","girl",
                    "Nymph","naruto","wolfman","chimera","monster","robot","ogre","skeleton","alien","zombie","shrek","santa",
                    "elder","Granny","baby",
                    "boxer","maid","guard","Wrestler","Magician","Scientist","pirate","Clown","Firefighter","cook","Pharaoh","cowboy","trollWarrior","villager","actor","swat","chef","captain","hero","mage","ninja","goalkeeper","viking","astronaut","worker","nurse","nun","farmer","doctor","warrior","butcher","knight","witch","wizard","pilot","racer","athlete","sportsman","police","policeman","driver","soldier",
                    "king","princess",
                    "groot","deadpool","spider-man","batman","ironman",
                    ],
            "idle;move":["snail","worm","spider","ant","Scorpion","locust",
                        "sloth","lizard",
                        "giraffe","Hippopotamus","Rhinoceros","Rhino","cow","donkey","llama","pig","Mammoth",
                        "truck","tank","car","van",]
        },
        "overwater":{
            "idle":[
                "lotus",
            ],
            "move;idle":[
                "ship","boat","dolphin","seal","whale","duck",
                "crab","Lobster","turtle",
            ]
            
        },
        "underwater":{
            "swim;idle":[
                "submarine",
                "dolphin","Anglerfish","fish","jellyfish","shark","penguin",
                "seal","whale",
                "crab","Lobster","octopus","turtle",
            ]
        }
    }
    
    
    


    
    @classmethod
    def create_validation_prompts(cls,**kwargs):
        
        data_root=kwargs["data_root"]
        label_root=kwargs["label_root"]
        total_mask_root=kwargs["mask_root"]
        
        seq_meta_data_map=kwargs["seq_meta_data_map"]
        
        seq_id_max_map=kwargs["seq_id_max_map"]
        hdri_json_data=kwargs["hdri_json_data"]
        
        
        allow_change_tgt=kwargs["allow_change_tgt"]
    
        asset_json_data=kwargs["asset_json_data"]
        
        
        cam_translation_rescale_factor=kwargs["cam_translation_rescale_factor"]
        
        obj_translation_rescale_factor=kwargs["obj_translation_rescale_factor"]
        
        mask_transforms=kwargs["mask_transforms"]
        tgt_fps_list=kwargs["tgt_fps_list"]
        ori_fps=kwargs["ori_fps"]
        time_duration=kwargs["time_duration"]
        
        mask_height=kwargs["height"]
        mask_width=kwargs["width"]
        
        
        while True:        
            data_type=random.choice(["single_static","multi_static","single_dynamic","multi_dynamic"])
            
            if "static" in data_type:
                static_type="static"
            else:
                static_type="dynamic"
                
            if "multi" in data_type:
                multi_suffix="_multi"
            else:
                multi_suffix=""
            seq_id_max=seq_id_max_map[data_type]
            if seq_id_max==0:
                continue
            seq_id=str(random.randint(0,seq_id_max))
            annotation_file_path=os.path.join(label_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}.json")
            clip_path=os.path.join(data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}")
            
            # seq_id=self.seq_id_list[idx]
            seq_meta_data=seq_meta_data_map[data_type][seq_id]
            
            if allow_change_tgt:
                tgt_fps=random.choice(tgt_fps_list)
                img_path_list,frame_idx_list, images = cls.sample_video_from_image_folder(
                    clip_path, ori_fps, time_duration, tgt_fps,  sample_num=16
                )
            else:
                # clip_time_list=self.get_clip_time_list(idx)
                clip_time_list=[]
                comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
                tgt_obj_id_list=eval(comment_dict["tgt_obj_id_list"])
                
                cam_time_range_list=eval(seq_meta_data["camera"]["Time_Range_List"])
                
                prev_tgt_id=None
                for time_range,tgt_id in zip(cam_time_range_list,tgt_obj_id_list):
                    if prev_tgt_id is None or tgt_id !=prev_tgt_id:
                        clip_time_list.append(time_range)
                    else:
                        assert clip_time_list[-1][-1]==time_range[0]
                        clip_time_list[-1][-1]=time_range[-1]
                        
                    prev_tgt_id=tgt_id
                    
                tgt_fps,img_path_list,frame_idx_list, images ,found= cls.sample_clip_from_image_folder(
                    clip_path, ori_fps, time_duration,clip_time_list,sample_num=16
                )
                if not found:
                    continue
                
            
            label_data={
                "annotation_file_path":annotation_file_path,
                "clip_path":clip_path,
            }
            
            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            

            
            background_description=get_background_description(hdri_json_data,seq_meta_data["camera"])

            
            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]

            _kwargs={
                "data_type":data_type,
                "seq_id":seq_id,
                "max_num":None,
            }
            
            seen_obj_id_list_list,seen_obj_idx_list_list,total_mask_list,obj_mask_list_list,object_description_list_list,action_description_list_list,action_type_list_list=[],[],[],[],[],[],[]
            for time_idx in frame_idx_list:
                # exr_file_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,"exr",f"{time_idx:04}.exr") ##traj_dataset/Rendered_Traj_Results
                # seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v2(exr_file_path,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**kwargs)
                mask_root=os.path.join(total_mask_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,str(time_idx))
                seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v3(mask_root,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**_kwargs)
                seen_obj_id_list_list.append(seen_obj_id_list)
                seen_obj_idx_list_list.append(seen_obj_idx_list)
                total_mask_list.append(total_mask)
                obj_mask_list_list.append(obj_mask_list)
                object_description_list_list.append(object_description_list)
                action_description_list_list.append(action_description_list)
                action_type_list_list.append(action_type_list)
                

            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            objs_dict=annotation_data["objects"]
        
            objs_info_np_list=[]
            for seen_obj_idx_list,time_idx in zip(seen_obj_idx_list_list,frame_idx_list):
                obj_info_list=[]
                for seen_obj_idx in seen_obj_idx_list:
                    seen_obj_idx=str(seen_obj_idx)
                    obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
                    obj_rot=transform_euler_to_matrix_v2(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
                    
                    ##transform to RealEstate-10K. [x,y,z] in realestate -> [y,-z,x] in unreal
                    
                    obj_rot=np.array(obj_rot)
                    # obj_rot=np.array([obj_rot[:,1],-obj_rot[:,2],obj_rot[:,0]])
                    
                    obj_info=np.eye(4)
                    
                    obj_info[:3,:3]=obj_rot
                    obj_info[:3,3]=np.array(obj_xyz)
                    obj_info_list.append(obj_info)
                    
                if obj_info_list:
                    objs_info_np_list.append(np.stack(obj_info_list))

            
            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]
            
            
            is_empty=any([len(seen_obj_id_list)==0 for seen_obj_id_list in seen_obj_id_list_list])
            
            obj_sentence_list_list,obj_adj_sentence_list_list,obj_no_adj_sentence_list_list,cam_sentence_list_list=[],[],[],[]
            if not is_empty:
                assert len(objs_info_np_list)==len(frame_idx_list)
                chosen_idx_list=[i for i in range(0, len(frame_idx_list), len(frame_idx_list)//3)][:4]
                
                is_track_single_object=True
                prev_obj_id = None
                for seen_obj_id_list in seen_obj_id_list_list:
                    if len(seen_obj_id_list)!=1:
                        is_track_single_object=False
                        break
                    if  prev_obj_id is not None and prev_obj_id!=seen_obj_id_list[0]:
                        is_track_single_object=False
                        break  
                    prev_obj_id=seen_obj_id_list[0]
                    

                if "multi" in data_type and not is_track_single_object:
                    prev_seen_obj_idx_list=[]
                
                    description_list=[]
                    seen_obj_idx_adj_description_map={}
                    # for chosen_seen_obj_idx_list,chosen_time_idx in zip(chosen_seen_obj_idx_list_list,chosen_time_idx_list):
                    for chosen_idx in chosen_idx_list:
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)    
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)
                        
                        for __idx,obj_idx in enumerate(chosen_seen_obj_idx_list):
                            if obj_idx not in seen_obj_idx_adj_description_map:
                                seen_obj_idx_adj_description_map[obj_idx]=obj_adj_sentence_list[__idx]
                                
                        
                        
                        if len(prev_seen_obj_idx_list)==0:
                            obj_num=len(chosen_seen_obj_idx_list)
                            use_back=random.choice([True,False])
                            if obj_num==1:
                                no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                if no_cam_des:
                                    sentence_list=obj_sentence_list
                                else:
                                    sentence_list=cam_sentence_list
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(first_sentence=sentence_list[0])
                                
                            else:
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                                        
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                        
                                    description=sentence_list[0]
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                            
                        else:
                            enter_obj_idx_list,exit_obj_idx_list=cls.get_enter_and_exit_obj_idx_list(prev_seen_obj_idx_list,chosen_seen_obj_idx_list)
                            
                            for _idx in enter_obj_idx_list:
                                assert _idx not in exit_obj_idx_list
                            
                            enter_obj_adj_sentence_list,exit_obj_adj_sentence_list=[],[]
                            for enter_obj_idx in enter_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(enter_obj_idx)
                                # assert _idx!=-1
                                enter_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[enter_obj_idx])
                                
                            for exit_obj_idx in exit_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(exit_obj_idx)
                                # assert _idx!=-1
                                exit_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[exit_obj_idx])
                            
                            if enter_obj_adj_sentence_list:
                                enter_description=enter_obj_adj_sentence_list[0]
                                
                                # temp_template="{sentence1_} {concat} {sentence_2}"
                                
                                
                                for enter_obj_adj_sentence in enter_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    enter_description=concat_template.format(sentence_1=enter_description,sentence_2=enter_obj_adj_sentence)
                                
                                enter_template=random.choice(UnrealTrajVideoDataset.ENTER_TEMPLATE)
                                enter_description=enter_template.format(objects=enter_description)
                            
                            
                            
                            if exit_obj_adj_sentence_list:
                                exit_description=exit_obj_adj_sentence_list[0]
                                
                                
                                
                                for exit_obj_adj_sentence in exit_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    exit_description=concat_template.format(sentence_1=exit_description,sentence_2=exit_obj_adj_sentence)
                                
                                exit_template=random.choice(UnrealTrajVideoDataset.EXIT_TEMPLATE)
                                exit_description=exit_template.format(objects=exit_description)
                                
                            
                            if enter_obj_adj_sentence_list and exit_obj_adj_sentence_list:
                                temp_template="{sentence_1} {concat} {sentence_2}"
                                flip=random.choice([True,False])
                                concat_str=random.choice(UnrealTrajVideoDataset.ENTER_EXIT_CONCAT_LIST)
                                if flip:
                                    description=temp_template.format(sentence_1=exit_description,concat=concat_str,sentence_2=enter_description)
                                else:
                                    description=temp_template.format(sentence_1=enter_description,concat=concat_str,sentence_2=exit_description)
                            
                            elif enter_obj_adj_sentence_list:
                                description=enter_description
                            elif exit_obj_adj_sentence_list:
                                description=exit_description
                            else:
                                description=""
                            
                        prev_seen_obj_idx_list=chosen_seen_obj_idx_list
                        
                        if description:
                            description_list.append(description)
                            
                    total_description=description_list[0]
                    
                    for description in description_list[1:]:
                        then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE)
                        total_description=then_template.format(sentence_1=total_description,sentence_2=description)
                    pass

                else:
                    action_type_change_list,action_type_change_t_idx_list,camera_type_change_list,camera_type_change_t_idx_list=[],[],[],[]
                    
                    inital_camera_type_list=None
                    prev_action_type,prev_camera_type=None,None
                    for t_idx,chosen_idx in enumerate(chosen_idx_list):
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        assert len(chosen_seen_obj_idx_list)==1 and len(chosen_object_description_list)==1 and len(chosen_action_description_list)==1 and len(chosen_action_type_list)==1
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)

                        if inital_camera_type_list is None:
                            initial_camera_type_list=camera_pose_description_list
                            
                        
                        if prev_action_type is not None:
                            if prev_action_type!=chosen_action_type_list[0]:
                                action_type_change_list.append(chosen_action_type_list[0])
                                action_type_change_t_idx_list.append(t_idx)
                                
                                
                            if prev_camera_type!=camera_pose_description_list[0]:
                                camera_type_change_list.append(camera_pose_description_list[0])
                                camera_type_change_t_idx_list.append(t_idx)
                                
                        prev_action_type=chosen_action_type_list[0]
                        prev_camera_type=camera_pose_description_list[0]
                        
                    
                    ##get action type change.
                    # use_action_change=random.choice([True,False])
                    use_action_change=random.choice([True,True])
                    obj_description=obj_sentence_list_list[0][0]
                    
                    use_back=random.choice([True,True])

                    if use_back:
                        ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(background=background_description,first_sentence=obj_description)
                    else:
                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(first_sentence=obj_description)
                        
                    if use_action_change:    
                        for action_change_t_idx in action_type_change_t_idx_list:
                            action_change_description=obj_no_adj_sentence_list_list[action_change_t_idx][0]
                            # is_replace_subject=random.uniform(0,1)
                            # if is_replace_subject <0.8:
                            #     subject_str=random.choice(UnrealTrajVideoDataset.SUBJECT_STR_LIST)
                            #     action_change_description.replace(object_description_list_list[0][0],subject_str)   
                            then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE) 
                            obj_description=then_template.format(sentence_1=obj_description,sentence_2=action_change_description)
                    
                    
                    total_description=obj_description

                
                use_descriptor_num=random.uniform(0,1)
                if use_descriptor_num<0.9:
                    use_descriptor=True
                else:
                    use_descriptor=False
                
                if use_descriptor:
                    descriptor_template=random.choice(UnrealTrajVideoDataset.DESCRIPTOR_TEMPLATE)
                    total_description=descriptor_template.format(sentence=total_description)      
            
            else:
                total_description=""
            
            if total_description!="":
                break
        
        
        
        
        circle_mask_list_list=[]
        
        use_sphere_mask=kwargs.get("use_sphere_mask",False)
        
        if use_sphere_mask:
            temp_save_dir="val_circle_mask"
            for time_idx,obj_mask_list in enumerate(obj_mask_list_list):
                circle_mask_list=[]
                for mask_idx,mask in enumerate(obj_mask_list):
                    # mask = np.resize(mask, (mask_height, mask_width, 1))
                    mask=(cv2.resize(mask.astype(np.uint8), (mask_width,mask_height), interpolation=cv2.INTER_NEAREST)>0.5)[...,None]
                    y, x = np.nonzero(mask[...,0])
                    gaussian_mask=mask[...,0]
                    if len(x) > 0 and len(y) > 0:
                        center, radius = cv2.minEnclosingCircle((np.column_stack((x, y))).astype(np.float32))
                        circle_mask = np.zeros_like(mask).astype(np.uint8)
                        # cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, 0)
                        # Draw a filled circle instead of just the outline
                        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
                        circle_mask=(circle_mask>0)
                        # Generate a Gaussian circle mask
                        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
                        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                        
                        # Create a Gaussian distribution
                        # sigma = radius / 3  # Adjust this value to control the softness of the edge
                        sigma = radius / 2  # Adjust this value to control the softness of the edge
                        # sigma = 10
                        # sigma = radius**2
                        gaussian_mask = np.exp(-0.5 * (dist_from_center / sigma)**2)
                        
                        
                        # Normalize the Gaussian mask
                        gaussian_mask = gaussian_mask / gaussian_mask.max()

                        
                        gaussian_mask=circle_mask[...,0]*gaussian_mask

                        # _circle_mask=gaussian_mask
                        
                        # img = Image.fromarray((_circle_mask*255).astype(np.uint8))

                        # img.save(os.path.join(temp_save_dir,f"{time_idx}-{mask_idx}.png"))
                        
                        # ord_mask=mask[...,0]
                        
                        # img = Image.fromarray((ord_mask).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"mask-{time_idx}-{mask_idx}.png"))
                        
                        
                        # img = Image.fromarray((circle_mask[...,0]).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"circle-mask-{time_idx}-{mask_idx}.png"))
                        # pass
                    circle_mask_list.append(torch.from_numpy(gaussian_mask[...,None]).permute(2, 0, 1).contiguous())
                
                # circle_mask_list_list.append(circle_mask_list)
                if circle_mask_list:
                    circle_mask_list_list.append(torch.stack(circle_mask_list))
                    
                    
        circle_mask_list=circle_mask_list_list
        
        
        
        new_obj_mask_list_list=[]
        for obj_mask_list in obj_mask_list_list:
            new_obj_mask_list=[]
            for obj_mask in obj_mask_list:
                new_obj_mask_list.append(torch.from_numpy(obj_mask).permute(2, 0, 1).contiguous())
            if new_obj_mask_list:
                new_obj_mask_list_list.append(torch.stack(new_obj_mask_list))
            else:
                assert False
            
        obj_mask_list=new_obj_mask_list_list
        
    
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(mask_transforms(obj_mask))
            # new_obj_mask_list_list.append(new_obj_mask_list)
        obj_mask_list=new_obj_mask_list
        
        
        
        new_circle_mask_list=[]
        for circle_mask in circle_mask_list:
            new_circle_mask_list.append(mask_transforms(circle_mask))
            # new_circle_mask_list_list.append(new_circle_mask_list)
        circle_mask_list=new_circle_mask_list
        
        
        objs_info_list=[]
        for obj_info in objs_info_np_list:
            objs_info_list.append(torch.from_numpy(obj_info))
        
        camera_info_np,intrinsics=cls.get_camera_info_np(label_data,frame_idx_list)
        
        camera_info=torch.from_numpy(camera_info_np)
        intrinsics=torch.from_numpy(intrinsics)
        
        
        
        rel_camera_info,rel_objs_info_list=cls.get_ref_matrix(camera_info,objs_info_list,cam_translation_rescale_factor,obj_translation_rescale_factor)
        
        
        
        first_frame_camera = camera_info[0]
        rotation_matrix = first_frame_camera[:3,:3]
        zero_translation = torch.zeros(3)
        new_first_camera_matrix = torch.zeros((3,4))
        new_first_camera_matrix[:3, :3] = rotation_matrix
        new_first_camera_matrix[:3, 3] = zero_translation
        
        # Flatten the new camera matrix to a 12-element vector
        new_first_camera_info = new_first_camera_matrix.reshape(-1)
        
        # Repeat this new camera info for all frames
        rel_camera_info[0]=new_first_camera_info
        
        return total_description,intrinsics,camera_info,rel_camera_info,objs_info_list,rel_objs_info_list,obj_mask_list,frame_idx_list,img_path_list,circle_mask_list,seen_obj_id_list_list
    
    
    
    @classmethod
    def create_validation_prompts_v2(cls,**kwargs):
        
        data_root=kwargs["data_root"]
        label_root=kwargs["label_root"]
        total_mask_root=kwargs["mask_root"]
        
        seq_meta_data_map=kwargs["seq_meta_data_map"]
        
        seq_id_max_map=kwargs["seq_id_max_map"]
        hdri_json_data=kwargs["hdri_json_data"]
        
        
        allow_change_tgt=kwargs["allow_change_tgt"]
    
        asset_json_data=kwargs["asset_json_data"]
        
        
        cam_translation_rescale_factor=kwargs["cam_translation_rescale_factor"]
        
        obj_translation_rescale_factor=kwargs["obj_translation_rescale_factor"]
        
        mask_transforms=kwargs["mask_transforms"]
        tgt_fps_list=kwargs["tgt_fps_list"]
        ori_fps=kwargs["ori_fps"]
        time_duration=kwargs["time_duration"]
        
        mask_height=kwargs["height"]
        mask_width=kwargs["width"]
        
        
        while True:        
            data_type=random.choice(["single_static","multi_static","single_dynamic","multi_dynamic"])
            
            if "static" in data_type:
                static_type="static"
            else:
                static_type="dynamic"
                
            if "multi" in data_type:
                multi_suffix="_multi"
            else:
                multi_suffix=""
            seq_id_max=seq_id_max_map[data_type]
            if seq_id_max==0:
                continue
            seq_id=str(random.randint(0,seq_id_max))
            annotation_file_path=os.path.join(label_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}.json")
            clip_path=os.path.join(data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}")
            
            # seq_id=self.seq_id_list[idx]
            seq_meta_data=seq_meta_data_map[data_type][seq_id]
            
            if allow_change_tgt:
                tgt_fps=random.choice(tgt_fps_list)
                img_path_list,frame_idx_list, images = cls.sample_video_from_image_folder(
                    clip_path, ori_fps, time_duration, tgt_fps,  sample_num=16
                )
            else:
                # clip_time_list=self.get_clip_time_list(idx)
                clip_time_list=[]
                comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
                tgt_obj_id_list=eval(comment_dict["tgt_obj_id_list"])
                
                cam_time_range_list=eval(seq_meta_data["camera"]["Time_Range_List"])
                
                prev_tgt_id=None
                for time_range,tgt_id in zip(cam_time_range_list,tgt_obj_id_list):
                    if prev_tgt_id is None or tgt_id !=prev_tgt_id:
                        clip_time_list.append(time_range)
                    else:
                        assert clip_time_list[-1][-1]==time_range[0]
                        clip_time_list[-1][-1]=time_range[-1]
                        
                    prev_tgt_id=tgt_id
                    
                tgt_fps,img_path_list,frame_idx_list, images ,found= cls.sample_clip_from_image_folder(
                    clip_path, ori_fps, time_duration,clip_time_list,sample_num=16
                )
                if not found:
                    continue
                
            
            label_data={
                "annotation_file_path":annotation_file_path,
                "clip_path":clip_path,
            }
            
            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            

            
            background_description=get_background_description(hdri_json_data,seq_meta_data["camera"])

            
            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]

            _kwargs={
                "data_type":data_type,
                "seq_id":seq_id,
                "max_num":None,
            }
            
            seen_obj_id_list_list,seen_obj_idx_list_list,total_mask_list,obj_mask_list_list,object_description_list_list,action_description_list_list,action_type_list_list=[],[],[],[],[],[],[]
            for time_idx in frame_idx_list:
                # exr_file_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,"exr",f"{time_idx:04}.exr") ##traj_dataset/Rendered_Traj_Results
                # seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v2(exr_file_path,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**kwargs)
                mask_root=os.path.join(total_mask_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,str(time_idx))
                seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v3(mask_root,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**_kwargs)
                seen_obj_id_list_list.append(seen_obj_id_list)
                seen_obj_idx_list_list.append(seen_obj_idx_list)
                total_mask_list.append(total_mask)
                obj_mask_list_list.append(obj_mask_list)
                object_description_list_list.append(object_description_list)
                action_description_list_list.append(action_description_list)
                action_type_list_list.append(action_type_list)
                

            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            objs_dict=annotation_data["objects"]
        
            objs_info_np_list=[]
            for seen_obj_idx_list,time_idx in zip(seen_obj_idx_list_list,frame_idx_list):
                obj_info_list=[]
                for seen_obj_idx in seen_obj_idx_list:
                    seen_obj_idx=str(seen_obj_idx)
                    obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
                    obj_rot=transform_euler_to_matrix_v2(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
                    
                    ##transform to RealEstate-10K. [x,y,z] in realestate -> [y,-z,x] in unreal
                    
                    obj_rot=np.array(obj_rot)
                    # obj_rot=np.array([obj_rot[:,1],-obj_rot[:,2],obj_rot[:,0]])
                    
                    obj_info=np.eye(4)
                    
                    obj_info[:3,:3]=obj_rot
                    obj_info[:3,3]=np.array(obj_xyz)
                    obj_info_list.append(obj_info)
                    
                if obj_info_list:
                    objs_info_np_list.append(np.stack(obj_info_list))
            

            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]
            
            
            is_empty=any([len(seen_obj_id_list)==0 for seen_obj_id_list in seen_obj_id_list_list])
            
            obj_sentence_list_list,obj_adj_sentence_list_list,obj_no_adj_sentence_list_list,cam_sentence_list_list=[],[],[],[]
            if not is_empty:
                assert len(objs_info_np_list)==len(frame_idx_list)
                chosen_idx_list=[i for i in range(0, len(frame_idx_list), len(frame_idx_list)//3)][:4]
                
                is_track_single_object=True
                prev_obj_id = None
                for seen_obj_id_list in seen_obj_id_list_list:
                    if len(seen_obj_id_list)!=1:
                        is_track_single_object=False
                        break
                    if  prev_obj_id is not None and prev_obj_id!=seen_obj_id_list[0]:
                        is_track_single_object=False
                        break  
                    prev_obj_id=seen_obj_id_list[0]
                    
                if "multi" in data_type and not is_track_single_object:
                    prev_seen_obj_idx_list=[]
                
                    description_list=[]
                    seen_obj_idx_adj_description_map={}
                    # for chosen_seen_obj_idx_list,chosen_time_idx in zip(chosen_seen_obj_idx_list_list,chosen_time_idx_list):
                    for chosen_idx in chosen_idx_list:
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)    
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)
                        
                        for __idx,obj_idx in enumerate(chosen_seen_obj_idx_list):
                            if obj_idx not in seen_obj_idx_adj_description_map:
                                seen_obj_idx_adj_description_map[obj_idx]=obj_adj_sentence_list[__idx]
                                
                        
                        
                        if len(prev_seen_obj_idx_list)==0:
                            obj_num=len(chosen_seen_obj_idx_list)
                            use_back=random.choice([True,False])
                            if obj_num==1:
                                no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                if no_cam_des:
                                    sentence_list=obj_sentence_list
                                else:
                                    sentence_list=cam_sentence_list
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(first_sentence=sentence_list[0])
                                
                            else:
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                                        
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                        
                                    description=sentence_list[0]
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                            
                        else:
                            enter_obj_idx_list,exit_obj_idx_list=cls.get_enter_and_exit_obj_idx_list(prev_seen_obj_idx_list,chosen_seen_obj_idx_list)
                            
                            for _idx in enter_obj_idx_list:
                                assert _idx not in exit_obj_idx_list
                            
                            enter_obj_adj_sentence_list,exit_obj_adj_sentence_list=[],[]
                            for enter_obj_idx in enter_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(enter_obj_idx)
                                # assert _idx!=-1
                                enter_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[enter_obj_idx])
                                
                            for exit_obj_idx in exit_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(exit_obj_idx)
                                # assert _idx!=-1
                                exit_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[exit_obj_idx])
                            
                            if enter_obj_adj_sentence_list:
                                enter_description=enter_obj_adj_sentence_list[0]
                                
                                # temp_template="{sentence1_} {concat} {sentence_2}"
                                
                                
                                for enter_obj_adj_sentence in enter_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    enter_description=concat_template.format(sentence_1=enter_description,sentence_2=enter_obj_adj_sentence)
                                
                                enter_template=random.choice(UnrealTrajVideoDataset.ENTER_TEMPLATE)
                                enter_description=enter_template.format(objects=enter_description)
                            
                            
                            
                            if exit_obj_adj_sentence_list:
                                exit_description=exit_obj_adj_sentence_list[0]
                                
                                
                                
                                for exit_obj_adj_sentence in exit_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    exit_description=concat_template.format(sentence_1=exit_description,sentence_2=exit_obj_adj_sentence)
                                
                                exit_template=random.choice(UnrealTrajVideoDataset.EXIT_TEMPLATE)
                                exit_description=exit_template.format(objects=exit_description)
                                
                            
                            if enter_obj_adj_sentence_list and exit_obj_adj_sentence_list:
                                temp_template="{sentence_1} {concat} {sentence_2}"
                                flip=random.choice([True,False])
                                concat_str=random.choice(UnrealTrajVideoDataset.ENTER_EXIT_CONCAT_LIST)
                                if flip:
                                    description=temp_template.format(sentence_1=exit_description,concat=concat_str,sentence_2=enter_description)
                                else:
                                    description=temp_template.format(sentence_1=enter_description,concat=concat_str,sentence_2=exit_description)
                            
                            elif enter_obj_adj_sentence_list:
                                description=enter_description
                            elif exit_obj_adj_sentence_list:
                                description=exit_description
                            else:
                                description=""
                            
                        prev_seen_obj_idx_list=chosen_seen_obj_idx_list
                        
                        if description:
                            description_list.append(description)
                            
                    total_description=description_list[0]
                    
                    for description in description_list[1:]:
                        then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE)
                        total_description=then_template.format(sentence_1=total_description,sentence_2=description)
                    pass

                else:
                    action_type_change_list,action_type_change_t_idx_list,camera_type_change_list,camera_type_change_t_idx_list=[],[],[],[]
                    
                    inital_camera_type_list=None
                    prev_action_type,prev_camera_type=None,None
                    for t_idx,chosen_idx in enumerate(chosen_idx_list):
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        assert len(chosen_seen_obj_idx_list)==1 and len(chosen_object_description_list)==1 and len(chosen_action_description_list)==1 and len(chosen_action_type_list)==1
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)

                        if inital_camera_type_list is None:
                            initial_camera_type_list=camera_pose_description_list
                            
                        
                        if prev_action_type is not None:
                            if prev_action_type!=chosen_action_type_list[0]:
                                action_type_change_list.append(chosen_action_type_list[0])
                                action_type_change_t_idx_list.append(t_idx)
                                
                                
                            if prev_camera_type!=camera_pose_description_list[0]:
                                camera_type_change_list.append(camera_pose_description_list[0])
                                camera_type_change_t_idx_list.append(t_idx)
                                
                        prev_action_type=chosen_action_type_list[0]
                        prev_camera_type=camera_pose_description_list[0]
                        
                    
                    ##get action type change.
                    # use_action_change=random.choice([True,False])
                    use_action_change=random.choice([True,True])
                    obj_description=obj_sentence_list_list[0][0]
                    
                    use_back=random.choice([True,True])

                    if use_back:
                        ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(background=background_description,first_sentence=obj_description)
                    else:
                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(first_sentence=obj_description)
                        
                    if use_action_change:    
                        for action_change_t_idx in action_type_change_t_idx_list:
                            action_change_description=obj_no_adj_sentence_list_list[action_change_t_idx][0]
                            # is_replace_subject=random.uniform(0,1)
                            # if is_replace_subject <0.8:
                            #     subject_str=random.choice(UnrealTrajVideoDataset.SUBJECT_STR_LIST)
                            #     action_change_description.replace(object_description_list_list[0][0],subject_str)   
                            then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE) 
                            obj_description=then_template.format(sentence_1=obj_description,sentence_2=action_change_description)
                    
                    
                    total_description=obj_description

                
                use_descriptor_num=random.uniform(0,1)
                if use_descriptor_num<0.9:
                    use_descriptor=True
                else:
                    use_descriptor=False
                
                if use_descriptor:
                    descriptor_template=random.choice(UnrealTrajVideoDataset.DESCRIPTOR_TEMPLATE)
                    total_description=descriptor_template.format(sentence=total_description)      
            
            else:
                total_description=""
            
            if total_description!="":
                break
        
        
        
        
        circle_mask_list_list=[]
        
        use_sphere_mask=kwargs.get("use_sphere_mask",False)
       
        if use_sphere_mask:
            # temp_save_dir="val_circle_mask"
            for time_idx,obj_mask_list in enumerate(obj_mask_list_list):
                circle_mask_list=[]
                for mask_idx,mask in enumerate(obj_mask_list):
                    # mask = np.resize(mask, (mask_height, mask_width, 1))
                    mask=(cv2.resize(mask.astype(np.uint8), (mask_width,mask_height), interpolation=cv2.INTER_NEAREST)>0.5)[...,None]
                    y, x = np.nonzero(mask[...,0])
                    gaussian_mask=mask[...,0]
                    if len(x) > 0 and len(y) > 0:
                        center, radius = cv2.minEnclosingCircle((np.column_stack((x, y))).astype(np.float32))
                        circle_mask = np.zeros_like(mask).astype(np.uint8)
                        # cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, 0)
                        # Draw a filled circle instead of just the outline
                        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
                        circle_mask=(circle_mask>0)
                        # Generate a Gaussian circle mask
                        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
                        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                        
                        # Create a Gaussian distribution
                        # sigma = radius / 3  # Adjust this value to control the softness of the edge
                        sigma = radius / 2  # Adjust this value to control the softness of the edge
                        # sigma = 10
                        # sigma = radius**2
                        gaussian_mask = np.exp(-0.5 * (dist_from_center / sigma)**2)
                        
                        
                        # Normalize the Gaussian mask
                        gaussian_mask = gaussian_mask / gaussian_mask.max()

                        
                        gaussian_mask=circle_mask[...,0]*gaussian_mask

                        # _circle_mask=gaussian_mask
                        
                        # img = Image.fromarray((_circle_mask*255).astype(np.uint8))

                        # img.save(os.path.join(temp_save_dir,f"{time_idx}-{mask_idx}.png"))
                        
                        # ord_mask=mask[...,0]
                        
                        # img = Image.fromarray((ord_mask).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"mask-{time_idx}-{mask_idx}.png"))
                        
                        
                        # img = Image.fromarray((circle_mask[...,0]).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"circle-mask-{time_idx}-{mask_idx}.png"))
                        # pass
                    circle_mask_list.append(torch.from_numpy(gaussian_mask[...,None]).permute(2, 0, 1).contiguous())
                
                # circle_mask_list_list.append(circle_mask_list)
                if circle_mask_list:
                    circle_mask_list_list.append(torch.stack(circle_mask_list))
                    
                    
        circle_mask_list=circle_mask_list_list
        
        
        
        new_obj_mask_list_list=[]
        for obj_mask_list in obj_mask_list_list:
            new_obj_mask_list=[]
            for obj_mask in obj_mask_list:
                new_obj_mask_list.append(torch.from_numpy(obj_mask).permute(2, 0, 1).contiguous())
            if new_obj_mask_list:
                new_obj_mask_list_list.append(torch.stack(new_obj_mask_list))
            else:
                assert False
            
        obj_mask_list=new_obj_mask_list_list
        
    
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(mask_transforms(obj_mask))
            # new_obj_mask_list_list.append(new_obj_mask_list)
        obj_mask_list=new_obj_mask_list
        
        
        
        new_circle_mask_list=[]
        for circle_mask in circle_mask_list:
            new_circle_mask_list.append(mask_transforms(circle_mask))

        circle_mask_list=new_circle_mask_list
        
        
        objs_info_list=[]
        for obj_info in objs_info_np_list:
            objs_info_list.append(torch.from_numpy(obj_info))
        
        camera_info_np,intrinsics=cls.get_camera_info_np(label_data,frame_idx_list)
        
        camera_info=torch.from_numpy(camera_info_np)
        intrinsics=torch.from_numpy(intrinsics)
        
        
        
        rel_camera_info,rel_objs_info_list=cls.get_ref_matrix(camera_info,objs_info_list,cam_translation_rescale_factor,obj_translation_rescale_factor)
        
        
        
        first_frame_camera = camera_info[0]
        rotation_matrix = first_frame_camera[:3,:3]
        zero_translation = torch.zeros(3)
        new_first_camera_matrix = torch.zeros((3,4))
        new_first_camera_matrix[:3, :3] = rotation_matrix
        new_first_camera_matrix[:3, 3] = zero_translation
        
        # Flatten the new camera matrix to a 12-element vector
        new_first_camera_info = new_first_camera_matrix.reshape(-1)
        
        # Repeat this new camera info for all frames
        rel_camera_info[0]=new_first_camera_info
        
        return total_description,intrinsics,camera_info,rel_camera_info,objs_info_list,rel_objs_info_list,obj_mask_list,frame_idx_list,img_path_list,circle_mask_list,seen_obj_id_list_list,scene_type,static_type
    
    @classmethod
    def create_validation_prompts_with_traj_change(cls,**kwargs):
        
        data_root=kwargs["data_root"]
        label_root=kwargs["label_root"]
        total_mask_root=kwargs["mask_root"]
        
        seq_meta_data_map=kwargs["seq_meta_data_map"]
        
        seq_id_max_map=kwargs["seq_id_max_map"]
        hdri_json_data=kwargs["hdri_json_data"]
        
        
        allow_change_tgt=kwargs["allow_change_tgt"]
    
        asset_json_data=kwargs["asset_json_data"]
        
        
        cam_translation_rescale_factor=kwargs["cam_translation_rescale_factor"]
        
        obj_translation_rescale_factor=kwargs["obj_translation_rescale_factor"]
        
        mask_transforms=kwargs["mask_transforms"]
        tgt_fps_list=kwargs["tgt_fps_list"]
        ori_fps=kwargs["ori_fps"]
        time_duration=kwargs["time_duration"]
        
        mask_height=kwargs["height"]
        mask_width=kwargs["width"]
        
        
        points_diff=kwargs["points_diff"]
        
        
        while True:        
            data_type=random.choice(["single_static","multi_static","single_dynamic","multi_dynamic"])
            
            if "static" in data_type:
                static_type="static"
            else:
                static_type="dynamic"
                
            if "multi" in data_type:
                multi_suffix="_multi"
            else:
                multi_suffix=""
            seq_id_max=seq_id_max_map[data_type]
            if seq_id_max==0:
                continue
            seq_id=str(random.randint(0,seq_id_max))
            annotation_file_path=os.path.join(label_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}.json")
            clip_path=os.path.join(data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{seq_id}")
            
            # seq_id=self.seq_id_list[idx]
            seq_meta_data=seq_meta_data_map[data_type][seq_id]
            
            if allow_change_tgt:
                tgt_fps=random.choice(tgt_fps_list)
                img_path_list,frame_idx_list, images = cls.sample_video_from_image_folder(
                    clip_path, ori_fps, time_duration, tgt_fps,  sample_num=16
                )
            else:
                # clip_time_list=self.get_clip_time_list(idx)
                clip_time_list=[]
                comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
                tgt_obj_id_list=eval(comment_dict["tgt_obj_id_list"])
                
                cam_time_range_list=eval(seq_meta_data["camera"]["Time_Range_List"])
                
                prev_tgt_id=None
                for time_range,tgt_id in zip(cam_time_range_list,tgt_obj_id_list):
                    if prev_tgt_id is None or tgt_id !=prev_tgt_id:
                        clip_time_list.append(time_range)
                    else:
                        assert clip_time_list[-1][-1]==time_range[0]
                        clip_time_list[-1][-1]=time_range[-1]
                        
                    prev_tgt_id=tgt_id
                    
                tgt_fps,img_path_list,frame_idx_list, images ,found= cls.sample_clip_from_image_folder(
                    clip_path, ori_fps, time_duration,clip_time_list,sample_num=16
                )
                if not found:
                    continue
                
            
            label_data={
                "annotation_file_path":annotation_file_path,
                "clip_path":clip_path,
            }
            
            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            

            
            background_description=get_background_description(hdri_json_data,seq_meta_data["camera"])

            
            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]

            _kwargs={
                "data_type":data_type,
                "seq_id":seq_id,
                "max_num":None,
            }
            
            seen_obj_id_list_list,seen_obj_idx_list_list,total_mask_list,obj_mask_list_list,object_description_list_list,action_description_list_list,action_type_list_list=[],[],[],[],[],[],[]
            for time_idx in frame_idx_list:
                # exr_file_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,"exr",f"{time_idx:04}.exr") ##traj_dataset/Rendered_Traj_Results
                # seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v2(exr_file_path,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**kwargs)
                mask_root=os.path.join(total_mask_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,str(time_idx))
                seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v3(mask_root,asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**_kwargs)
                seen_obj_id_list_list.append(seen_obj_id_list)
                seen_obj_idx_list_list.append(seen_obj_idx_list)
                total_mask_list.append(total_mask)
                obj_mask_list_list.append(obj_mask_list)
                object_description_list_list.append(object_description_list)
                action_description_list_list.append(action_description_list)
                action_type_list_list.append(action_type_list)
                

            annotation_file_path=label_data["annotation_file_path"]
            with open(annotation_file_path,"r") as f:
                annotation_data=json.load(f)
            objs_dict=annotation_data["objects"]
        
            objs_info_np_list=[]

            for seen_obj_idx_list,time_idx in zip(seen_obj_idx_list_list,frame_idx_list):
                obj_info_list=[]
                for seen_obj_idx in seen_obj_idx_list:
                    seen_obj_idx=str(seen_obj_idx)
                    obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
                    obj_rot=transform_euler_to_matrix_v2(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
                    
                    ##transform to RealEstate-10K. [x,y,z] in realestate -> [y,-z,x] in unreal
                    
                    obj_rot=np.array(obj_rot)
                    # obj_rot=np.array([obj_rot[:,1],-obj_rot[:,2],obj_rot[:,0]])
                    
                    obj_info=np.eye(4)
                    
                    obj_info[:3,:3]=obj_rot
                    obj_info[:3,3]=np.array(obj_xyz)
                    obj_info_list.append(obj_info)
                    
                if obj_info_list:
                    objs_info_np_list.append(np.stack(obj_info_list))
            
            
            comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
            scene_type=comment_dict["scene_type"]
            
            
            is_empty=any([len(seen_obj_id_list)==0 for seen_obj_id_list in seen_obj_id_list_list])
            
            obj_sentence_list_list,obj_adj_sentence_list_list,obj_no_adj_sentence_list_list,cam_sentence_list_list=[],[],[],[]
            if not is_empty:
                assert len(objs_info_np_list)==len(frame_idx_list)
                chosen_idx_list=[i for i in range(0, len(frame_idx_list), len(frame_idx_list)//3)][:4]
                
                is_track_single_object=True
                prev_obj_id = None
                for seen_obj_id_list in seen_obj_id_list_list:
                    if len(seen_obj_id_list)!=1:
                        is_track_single_object=False
                        break
                    if  prev_obj_id is not None and prev_obj_id!=seen_obj_id_list[0]:
                        is_track_single_object=False
                        break  
                    prev_obj_id=seen_obj_id_list[0]
                
                
                assert is_track_single_object
                    
                if "multi" in data_type and not is_track_single_object:
                    prev_seen_obj_idx_list=[]
                
                    description_list=[]
                    seen_obj_idx_adj_description_map={}
                    # for chosen_seen_obj_idx_list,chosen_time_idx in zip(chosen_seen_obj_idx_list_list,chosen_time_idx_list):
                    for chosen_idx in chosen_idx_list:
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)    
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)
                        
                        for __idx,obj_idx in enumerate(chosen_seen_obj_idx_list):
                            if obj_idx not in seen_obj_idx_adj_description_map:
                                seen_obj_idx_adj_description_map[obj_idx]=obj_adj_sentence_list[__idx]
                                
                        
                        
                        if len(prev_seen_obj_idx_list)==0:
                            obj_num=len(chosen_seen_obj_idx_list)
                            use_back=random.choice([True,False])
                            if obj_num==1:
                                no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                if no_cam_des:
                                    sentence_list=obj_sentence_list
                                else:
                                    sentence_list=cam_sentence_list
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    description=ass_template.format(first_sentence=sentence_list[0])
                                
                            else:
                                if use_back:
                                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                    description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                                        
                                else:
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                        
                                    description=sentence_list[0]
                                    for sentence_idx in range(1,len(sentence_list)):
                                        no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                        if no_cam_des:
                                            sentence_list=obj_sentence_list
                                        else:
                                            sentence_list=cam_sentence_list
                                        
                                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                        ass_template=random.choice(ass_template_list)
                                        concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                        description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                            
                        else:
                            enter_obj_idx_list,exit_obj_idx_list=cls.get_enter_and_exit_obj_idx_list(prev_seen_obj_idx_list,chosen_seen_obj_idx_list)
                            
                            for _idx in enter_obj_idx_list:
                                assert _idx not in exit_obj_idx_list
                            
                            enter_obj_adj_sentence_list,exit_obj_adj_sentence_list=[],[]
                            for enter_obj_idx in enter_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(enter_obj_idx)
                                # assert _idx!=-1
                                enter_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[enter_obj_idx])
                                
                            for exit_obj_idx in exit_obj_idx_list:
                                # _idx=seen_obj_idx_list.index(exit_obj_idx)
                                # assert _idx!=-1
                                exit_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[exit_obj_idx])
                            
                            if enter_obj_adj_sentence_list:
                                enter_description=enter_obj_adj_sentence_list[0]
                                
                                # temp_template="{sentence1_} {concat} {sentence_2}"
                                
                                
                                for enter_obj_adj_sentence in enter_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    enter_description=concat_template.format(sentence_1=enter_description,sentence_2=enter_obj_adj_sentence)
                                
                                enter_template=random.choice(UnrealTrajVideoDataset.ENTER_TEMPLATE)
                                enter_description=enter_template.format(objects=enter_description)
                            
                            
                            
                            if exit_obj_adj_sentence_list:
                                exit_description=exit_obj_adj_sentence_list[0]
                                
                                
                                
                                for exit_obj_adj_sentence in exit_obj_adj_sentence_list[1:]:
                                    concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                    exit_description=concat_template.format(sentence_1=exit_description,sentence_2=exit_obj_adj_sentence)
                                
                                exit_template=random.choice(UnrealTrajVideoDataset.EXIT_TEMPLATE)
                                exit_description=exit_template.format(objects=exit_description)
                                
                            
                            if enter_obj_adj_sentence_list and exit_obj_adj_sentence_list:
                                temp_template="{sentence_1} {concat} {sentence_2}"
                                flip=random.choice([True,False])
                                concat_str=random.choice(UnrealTrajVideoDataset.ENTER_EXIT_CONCAT_LIST)
                                if flip:
                                    description=temp_template.format(sentence_1=exit_description,concat=concat_str,sentence_2=enter_description)
                                else:
                                    description=temp_template.format(sentence_1=enter_description,concat=concat_str,sentence_2=exit_description)
                            
                            elif enter_obj_adj_sentence_list:
                                description=enter_description
                            elif exit_obj_adj_sentence_list:
                                description=exit_description
                            else:
                                description=""
                            
                        prev_seen_obj_idx_list=chosen_seen_obj_idx_list
                        
                        if description:
                            description_list.append(description)
                            
                    total_description=description_list[0]
                    
                    for description in description_list[1:]:
                        then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE)
                        total_description=then_template.format(sentence_1=total_description,sentence_2=description)
                    pass

                else:
                    action_type_change_list,action_type_change_t_idx_list,camera_type_change_list,camera_type_change_t_idx_list=[],[],[],[]
                    
                    inital_camera_type_list=None
                    prev_action_type,prev_camera_type=None,None
                    for t_idx,chosen_idx in enumerate(chosen_idx_list):
                        chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                        assert len(chosen_seen_obj_idx_list)==1 and len(chosen_object_description_list)==1 and len(chosen_action_description_list)==1 and len(chosen_action_type_list)==1
                        camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)
                        obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=cls.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
                
                        obj_sentence_list_list.append(obj_sentence_list)
                        obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                        obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                        cam_sentence_list_list.append(cam_sentence_list)

                        if inital_camera_type_list is None:
                            initial_camera_type_list=camera_pose_description_list
                            
                        
                        if prev_action_type is not None:
                            if prev_action_type!=chosen_action_type_list[0]:
                                action_type_change_list.append(chosen_action_type_list[0])
                                action_type_change_t_idx_list.append(t_idx)
                                
                                
                            if prev_camera_type!=camera_pose_description_list[0]:
                                camera_type_change_list.append(camera_pose_description_list[0])
                                camera_type_change_t_idx_list.append(t_idx)
                                
                        prev_action_type=chosen_action_type_list[0]
                        prev_camera_type=camera_pose_description_list[0]
                        
                    
                    ##get action type change.
                    # use_action_change=random.choice([True,False])
                    use_action_change=random.choice([True,True])
                    obj_description=obj_sentence_list_list[0][0]
                    
                    use_back=random.choice([True,True])

                    if use_back:
                        ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(background=background_description,first_sentence=obj_description)
                    else:
                        ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                        ass_template=random.choice(ass_template_list)
                        obj_description=ass_template.format(first_sentence=obj_description)
                        
                    if use_action_change:    
                        for action_change_t_idx in action_type_change_t_idx_list:
                            action_change_description=obj_no_adj_sentence_list_list[action_change_t_idx][0] 
                            then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE) 
                            obj_description=then_template.format(sentence_1=obj_description,sentence_2=action_change_description)
                    
                    
                    total_description=obj_description

                
                use_descriptor_num=random.uniform(0,1)
                if use_descriptor_num<0.9:
                    use_descriptor=True
                else:
                    use_descriptor=False
                
                if use_descriptor:
                    descriptor_template=random.choice(UnrealTrajVideoDataset.DESCRIPTOR_TEMPLATE)
                    total_description=descriptor_template.format(sentence=total_description)      
            
            else:
                total_description=""
            
            if total_description!="":
                break
        
        
        
        
        circle_mask_list_list=[]
        
        use_sphere_mask=kwargs.get("use_sphere_mask",False)

        
        if use_sphere_mask:
            temp_save_dir="mask_offset"
            
            
            first_point=None
            for time_idx,obj_mask_list in enumerate(obj_mask_list_list):
                circle_mask_list=[]
                point_diff=points_diff[time_idx] 
                
                
                for mask_idx,mask in enumerate(obj_mask_list):
                    # mask = np.resize(mask, (mask_height, mask_width, 1))
                    mask=(cv2.resize(mask.astype(np.uint8), (mask_width,mask_height), interpolation=cv2.INTER_NEAREST)>0.5)[...,None]
                    y, x = np.nonzero(mask[...,0])
                    gaussian_mask=mask[...,0]
                    if len(x) > 0 and len(y) > 0:
                        center, radius = cv2.minEnclosingCircle((np.column_stack((x, y))).astype(np.float32))
                        circle_mask = np.zeros_like(mask).astype(np.uint8)

                        if first_point is None:
                            first_point=center
                            
                        new_center=[]
                        
                        new_center.append(max(min(first_point[0]+point_diff[0],mask_width),0))
                        new_center.append(max(min(first_point[1]+point_diff[1],mask_height),0))
                        
                        
                        # center=[ min(ord_xy+off_xy) ]
                        center=new_center
                        
                        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
                        circle_mask=(circle_mask>0)
                        # Generate a Gaussian circle mask
                        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
                        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                        
                        # Create a Gaussian distribution
                        # sigma = radius / 3  # Adjust this value to control the softness of the edge
                        sigma = radius / 2  # Adjust this value to control the softness of the edge
                        # sigma = 10
                        # sigma = radius**2
                        gaussian_mask = np.exp(-0.5 * (dist_from_center / sigma)**2)
                        
                        
                        # Normalize the Gaussian mask
                        gaussian_mask = gaussian_mask / gaussian_mask.max()

                        
                        gaussian_mask=circle_mask[...,0]*gaussian_mask

                        # _circle_mask=gaussian_mask
                        
                        # img = Image.fromarray((_circle_mask*255).astype(np.uint8))

                        # img.save(os.path.join(temp_save_dir,f"{time_idx}-{mask_idx}.png"))
                        
                        # ord_mask=mask[...,0]
                        
                        # img = Image.fromarray((ord_mask).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"mask-{time_idx}-{mask_idx}.png"))
                        
                        
                        # img = Image.fromarray((circle_mask[...,0]).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"circle-mask-{time_idx}-{mask_idx}.png"))
                        # pass
                    circle_mask_list.append(torch.from_numpy(gaussian_mask[...,None]).permute(2, 0, 1).contiguous())
                
                # circle_mask_list_list.append(circle_mask_list)
                if circle_mask_list:
                    circle_mask_list_list.append(torch.stack(circle_mask_list))
                    
                    
        circle_mask_list=circle_mask_list_list
        
        
        
        new_obj_mask_list_list=[]
        for obj_mask_list in obj_mask_list_list:
            new_obj_mask_list=[]
            for obj_mask in obj_mask_list:
                new_obj_mask_list.append(torch.from_numpy(obj_mask).permute(2, 0, 1).contiguous())
            if new_obj_mask_list:
                new_obj_mask_list_list.append(torch.stack(new_obj_mask_list))
            else:
                assert False
            
        obj_mask_list=new_obj_mask_list_list
        
    
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(mask_transforms(obj_mask))
            # new_obj_mask_list_list.append(new_obj_mask_list)
        obj_mask_list=new_obj_mask_list
        
        
        
        new_circle_mask_list=[]
        for circle_mask in circle_mask_list:
            new_circle_mask_list.append(mask_transforms(circle_mask))
            # new_circle_mask_list_list.append(new_circle_mask_list)
        circle_mask_list=new_circle_mask_list
        
        
        objs_info_list=[]
        for obj_info in objs_info_np_list:
            objs_info_list.append(torch.from_numpy(obj_info))
        
        camera_info_np,intrinsics=cls.get_camera_info_np(label_data,frame_idx_list)
        
        camera_info=torch.from_numpy(camera_info_np)
        intrinsics=torch.from_numpy(intrinsics)
        
        
        
        rel_camera_info,rel_objs_info_list=cls.get_ref_matrix(camera_info,objs_info_list,cam_translation_rescale_factor,obj_translation_rescale_factor)
        
        
        
        first_frame_camera = camera_info[0]
        rotation_matrix = first_frame_camera[:3,:3]
        zero_translation = torch.zeros(3)
        new_first_camera_matrix = torch.zeros((3,4))
        new_first_camera_matrix[:3, :3] = rotation_matrix
        new_first_camera_matrix[:3, 3] = zero_translation
        
        # Flatten the new camera matrix to a 12-element vector
        new_first_camera_info = new_first_camera_matrix.reshape(-1)
        
        # Repeat this new camera info for all frames
        rel_camera_info[0]=new_first_camera_info
        
        return total_description,intrinsics,camera_info,rel_camera_info,objs_info_list,rel_objs_info_list,obj_mask_list,frame_idx_list,img_path_list,circle_mask_list,seen_obj_id_list_list,scene_type,static_type
    
    
    def __init__(
        self,
        # root_path,
        sample_n_frames,
        ori_fps,
        time_duration,
        tgt_fps_list,
        
        allow_change_tgt,

        
        data_root,
        lable_root,
        mask_root,
        seq_csv_root,
        
        hdri_json_file_path,
        asset_json_file_path,

        tokenizer=None,
        single_static_num=6000,
        single_dynamic_num=8000,
        multi_static_num=6000,
        multi_dynamic_num=6000,
        # sample_stride=4,
        sample_size=[256, 384],
        is_image=False,
        use_flip=True,
        cam_translation_rescale_factor=1,
        obj_translation_rescale_factor=1,
        use_sphere_mask=False,
    ):
        # self.root_path = root_path
        self.tokenizer=tokenizer
        self.cam_translation_rescale_factor=cam_translation_rescale_factor
        self.obj_translation_rescale_factor=obj_translation_rescale_factor
        
        self.ori_fps=ori_fps
        self.time_duration=time_duration
        
        self.tgt_fps_list=tgt_fps_list

        self.data_root=data_root
        self.lable_root=lable_root
        self.mask_root=mask_root
        
        self.allow_change_tgt= allow_change_tgt
        
        self.use_sphere_mask=use_sphere_mask
        
        self.seq_csv_root=seq_csv_root
        self.hdri_json_file_path=hdri_json_file_path
        
        with open(hdri_json_file_path,"r") as f:
            self.hdri_json_data=json.load(f)
        
        with open(asset_json_file_path,"r") as f:
            self.asset_json_data=json.load(f)
        self.asset_json_file_path=asset_json_file_path
    
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        self.single_static_num=single_static_num
        self.single_dynamic_num=single_dynamic_num
        self.multi_static_num=multi_static_num
        self.multi_dynamic_num=multi_dynamic_num

        self.length=0
        self.dataset=[]
        self.data_type_list=[]
        self.seq_id_list=[]

        
        for single_type in ["single","multi"]:
            for static_type in ["static","dynamic"]: ##format of key is {single/multi}_{static/dynamic}
                if single_type=="single" and static_type=="static":
                    num=self.single_static_num
                    static_type="static"
                    multi_suffix=""
                
                if single_type=="single" and static_type=="dynamic":
                    num=self.single_dynamic_num
                    static_type="dynamic"
                    multi_suffix=""
                
                if single_type=="multi" and static_type=="static":
                    num=self.multi_static_num
                    static_type="static"
                    multi_suffix="_multi"
                    
                if single_type=="multi" and static_type=="dynamic":
                    num=self.multi_dynamic_num
                    static_type="dynamic"
                    multi_suffix="_multi"
                    
                for i in range(num):
                    annotation_file_path=os.path.join(self.lable_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{i}.json")
                    clip_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,f"{i}")
                    self.dataset.append({
                        "annotation_file_path":annotation_file_path,
                        "clip_path":clip_path,
                    })
                    self.data_type_list.append("_".join([single_type,static_type]))
                    self.seq_id_list.append(str(i))
                
        self.length=len(self.dataset)
            
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        if use_flip:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            
            mask_transforms = [transforms.Resize(sample_size),
                    transforms.RandomHorizontalFlip(),
                    ]
        else:
            pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            
            mask_transforms = [transforms.Resize(sample_size)
            ]
            
        self.pixel_transforms = transforms.Compose(pixel_transforms)
        
        self.mask_transforms = transforms.Compose(mask_transforms)

        self.seq_meta_data_map=self._get_csv_meta_data_map()
    
    
    def _get_csv_meta_data_map(self):
        seq_meta_data_map={}
        for dynamic_type_str in ["static","dynamic"]:
            for single_type_str in ["","_multi"]:
                seq_meta_data={}
                csv_path=os.path.join(self.seq_csv_root,f"traj_{dynamic_type_str}{single_type_str}.csv")
                with open(csv_path, mode="r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    csv_rows = list(
                        csv_reader
                    )  # Convert to list of rows so that we can look ahead, this will skip header

                for row_index, row in enumerate(csv_rows):
                    if row["Type"] == "Group":
                        # start_index = row_index
                        seq_id = row["Seq_ID"]
                        body_id=-1
                        row.pop("Seq_ID")
                        seq_meta_data[seq_id]={}
                        seq_meta_data[seq_id]["camera"]=row
                        seq_meta_data[seq_id]["objects"]={}
                    else:
                        body_id+=1
                        row.pop("Seq_ID")
                        seq_meta_data[seq_id]["objects"][str(body_id)]=row
                               
                if dynamic_type_str=="static":
                    dynamic_type="static"
                else:
                    dynamic_type="dynamic"
                    
                if single_type_str=="_multi":
                    single_type="multi"
                else:
                    single_type="single"
                key_name=f"{single_type}_{dynamic_type}"
                
                seq_meta_data_map[key_name]=seq_meta_data
            

        return seq_meta_data_map


    def load_img_list(self, idx):
        video_dict = self.dataset[idx]
        video_path=video_dict['clip_path']
        frame_files = sorted(
            [
                os.path.join(video_path, f)
                for f in os.listdir(video_path)
                if os.path.isfile(os.path.join(video_path, f)) and f.endswith(".png") and "-" not in f
            ]
        )
        return frame_files

    def get_text_prompt_and_mask_list(self,idx,frame_idx_list):
        data_type=self.data_type_list[idx]
        label_data=self.dataset[idx]
        seq_id=self.seq_id_list[idx]
        annotation_file_path=label_data["annotation_file_path"]
        with open(annotation_file_path,"r") as f:
            annotation_data=json.load(f)
        
        # seq_id=self.seq_id_list[idx]
        seq_meta_data=self.seq_meta_data_map[data_type][seq_id]
        
        background_description=get_background_description(self.hdri_json_data,seq_meta_data["camera"])
        if "static" in data_type:
            static_type="static"
        else:
            static_type="dynamic"
            
        if "multi" in data_type:
            multi_suffix="_multi"
        else:
            multi_suffix=""
        
        data_type=self.data_type_list[idx]
        # label_data=self.dataset[idx]
        seq_id=self.seq_id_list[idx]
        kwargs={
            "data_type":data_type,
            "seq_id":seq_id,
            "max_num":None,
        }
        
        seen_obj_id_list_list,seen_obj_idx_list_list,total_mask_list,obj_mask_list_list,object_description_list_list,action_description_list_list,action_type_list_list=[],[],[],[],[],[],[]
        for time_idx in frame_idx_list:
            # exr_file_path=os.path.join(self.data_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,"exr",f"{time_idx:04}.exr") ##traj_dataset/Rendered_Traj_Results
            # seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v2(exr_file_path,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**kwargs)
            mask_root=os.path.join(self.mask_root,f"Rendered_Traj_Results{multi_suffix}",static_type,seq_id,str(time_idx))
            seen_obj_id_list,seen_obj_idx_list,total_mask,obj_mask_list,object_description_list,action_description_list,action_type_list=get_seen_object_and_action_description_v3(mask_root,self.asset_json_data,seq_meta_data,time_idx,appearance_percentage=0.0015,**kwargs)
            seen_obj_id_list_list.append(seen_obj_id_list)
            seen_obj_idx_list_list.append(seen_obj_idx_list)
            total_mask_list.append(total_mask)
            obj_mask_list_list.append(obj_mask_list)
            object_description_list_list.append(object_description_list)
            action_description_list_list.append(action_description_list)
            action_type_list_list.append(action_type_list)
            

        annotation_file_path=label_data["annotation_file_path"]
        with open(annotation_file_path,"r") as f:
            annotation_data=json.load(f)
        objs_dict=annotation_data["objects"]
    
        objs_info_np_list=[]
        for seen_obj_idx_list,time_idx in zip(seen_obj_idx_list_list,frame_idx_list):
            obj_info_list=[]
            for seen_obj_idx in seen_obj_idx_list:
                seen_obj_idx=str(seen_obj_idx)
                obj_xyz,obj_euler_rot=objs_dict[seen_obj_idx][time_idx][-3:],objs_dict[seen_obj_idx][time_idx][3:6]
                obj_rot=transform_euler_to_matrix_v2(obj_euler_rot[2],obj_euler_rot[1],obj_euler_rot[0]) ##,roll, pitch , yaw
                
                ##transform to RealEstate-10K. [x,y,z] in realestate -> [y,-z,x] in unreal
                
                obj_rot=np.array(obj_rot)
                # obj_rot=np.array([obj_rot[:,1],-obj_rot[:,2],obj_rot[:,0]])
                
                obj_info=np.eye(4)
                
                obj_info[:3,:3]=obj_rot
                obj_info[:3,3]=np.array(obj_xyz)
                obj_info_list.append(obj_info)
                
            if obj_info_list:
                objs_info_np_list.append(np.stack(obj_info_list))
        
        # mask_root="temp_mask_motion"
        
        comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
        scene_type=comment_dict["scene_type"]
        
        
        is_empty=any([len(seen_obj_id_list)==0 for seen_obj_id_list in seen_obj_id_list_list])
        
        obj_sentence_list_list,obj_adj_sentence_list_list,obj_no_adj_sentence_list_list,cam_sentence_list_list=[],[],[],[]
        if not is_empty:
            assert len(objs_info_np_list)==len(frame_idx_list)
            chosen_idx_list=[i for i in range(0, len(frame_idx_list), len(frame_idx_list)//3)][:4]
            
            is_track_single_object=True
            prev_obj_id = None
            for seen_obj_id_list in seen_obj_id_list_list:
                if len(seen_obj_id_list)!=1:
                    is_track_single_object=False
                    break
                if  prev_obj_id is not None and prev_obj_id!=seen_obj_id_list[0]:
                    is_track_single_object=False
                    break  
                prev_obj_id=seen_obj_id_list[0]
                
            if "multi" in data_type and not is_track_single_object:
                prev_seen_obj_idx_list=[]
            
                description_list=[]
                seen_obj_idx_adj_description_map={}
                # for chosen_seen_obj_idx_list,chosen_time_idx in zip(chosen_seen_obj_idx_list_list,chosen_time_idx_list):
                for chosen_idx in chosen_idx_list:
                    chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                    
                    camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)    
                    obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=self.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
            
                    obj_sentence_list_list.append(obj_sentence_list)
                    obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                    obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                    cam_sentence_list_list.append(cam_sentence_list)
                    
                    for __idx,obj_idx in enumerate(chosen_seen_obj_idx_list):
                        if obj_idx not in seen_obj_idx_adj_description_map:
                            seen_obj_idx_adj_description_map[obj_idx]=obj_adj_sentence_list[__idx]
                            
                    
                    
                    if len(prev_seen_obj_idx_list)==0:
                        obj_num=len(chosen_seen_obj_idx_list)
                        use_back=random.choice([True,False])
                        if obj_num==1:
                            no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                            if no_cam_des:
                                sentence_list=obj_sentence_list
                            else:
                                sentence_list=cam_sentence_list
                            if use_back:
                                ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                ass_template=random.choice(ass_template_list)
                                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                            else:
                                ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                                ass_template=random.choice(ass_template_list)
                                description=ass_template.format(first_sentence=sentence_list[0])
                            
                        else:
                            if use_back:
                                ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                                ass_template=random.choice(ass_template_list)
                                no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                if no_cam_des:
                                    sentence_list=obj_sentence_list
                                else:
                                    sentence_list=cam_sentence_list
                                description=ass_template.format(background=background_description,first_sentence=sentence_list[0])
                                for sentence_idx in range(1,len(sentence_list)):
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                                    
                            else:
                                ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                ass_template=random.choice(ass_template_list)
                                no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                if no_cam_des:
                                    sentence_list=obj_sentence_list
                                else:
                                    sentence_list=cam_sentence_list
                                    
                                description=sentence_list[0]
                                for sentence_idx in range(1,len(sentence_list)):
                                    no_cam_des=random.choice([True,True])##no_cam_des=random.choice([True,False])
                                    if no_cam_des:
                                        sentence_list=obj_sentence_list
                                    else:
                                        sentence_list=cam_sentence_list
                                    
                                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_MULTI_TEMPLATE
                                    ass_template=random.choice(ass_template_list)
                                    concat_str=random.choice(UnrealTrajVideoDataset.CONCAT_LIST)
                                    description=ass_template.format(first_sentence=description,concat=concat_str,second_sentence=sentence_list[sentence_idx])
                        
                    else:
                        enter_obj_idx_list,exit_obj_idx_list=self.get_enter_and_exit_obj_idx_list(prev_seen_obj_idx_list,chosen_seen_obj_idx_list)
                        
                        for _idx in enter_obj_idx_list:
                            assert _idx not in exit_obj_idx_list
                        
                        enter_obj_adj_sentence_list,exit_obj_adj_sentence_list=[],[]
                        for enter_obj_idx in enter_obj_idx_list:
                            # _idx=seen_obj_idx_list.index(enter_obj_idx)
                            # assert _idx!=-1
                            enter_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[enter_obj_idx])
                            
                        for exit_obj_idx in exit_obj_idx_list:
                            # _idx=seen_obj_idx_list.index(exit_obj_idx)
                            # assert _idx!=-1
                            exit_obj_adj_sentence_list.append(seen_obj_idx_adj_description_map[exit_obj_idx])
                        
                        if enter_obj_adj_sentence_list:
                            enter_description=enter_obj_adj_sentence_list[0]
                            
                            # temp_template="{sentence1_} {concat} {sentence_2}"
                            
                            
                            for enter_obj_adj_sentence in enter_obj_adj_sentence_list[1:]:
                                concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                enter_description=concat_template.format(sentence_1=enter_description,sentence_2=enter_obj_adj_sentence)
                            
                            enter_template=random.choice(UnrealTrajVideoDataset.ENTER_TEMPLATE)
                            enter_description=enter_template.format(objects=enter_description)
                        
                        
                        
                        if exit_obj_adj_sentence_list:
                            exit_description=exit_obj_adj_sentence_list[0]
                            
                            
                            
                            for exit_obj_adj_sentence in exit_obj_adj_sentence_list[1:]:
                                concat_template=random.choice(UnrealTrajVideoDataset.OBJ_CONCAT_LIST)
                                exit_description=concat_template.format(sentence_1=exit_description,sentence_2=exit_obj_adj_sentence)
                            
                            exit_template=random.choice(UnrealTrajVideoDataset.EXIT_TEMPLATE)
                            exit_description=exit_template.format(objects=exit_description)
                            
                        
                        if enter_obj_adj_sentence_list and exit_obj_adj_sentence_list:
                            temp_template="{sentence_1} {concat} {sentence_2}"
                            flip=random.choice([True,False])
                            concat_str=random.choice(UnrealTrajVideoDataset.ENTER_EXIT_CONCAT_LIST)
                            if flip:
                                description=temp_template.format(sentence_1=exit_description,concat=concat_str,sentence_2=enter_description)
                            else:
                                description=temp_template.format(sentence_1=enter_description,concat=concat_str,sentence_2=exit_description)
                        
                        elif enter_obj_adj_sentence_list:
                            description=enter_description
                        elif exit_obj_adj_sentence_list:
                            description=exit_description
                        else:
                            description=""
                        
                    prev_seen_obj_idx_list=chosen_seen_obj_idx_list
                    
                    if description:
                        description_list.append(description)
                        
                total_description=description_list[0]
                
                for description in description_list[1:]:
                    then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE)
                    total_description=then_template.format(sentence_1=total_description,sentence_2=description)
                pass

            else:
                action_type_change_list,action_type_change_t_idx_list,camera_type_change_list,camera_type_change_t_idx_list=[],[],[],[]
                
                inital_camera_type_list=None
                prev_action_type,prev_camera_type=None,None
                for t_idx,chosen_idx in enumerate(chosen_idx_list):
                    chosen_seen_obj_idx_list,chosen_time_idx,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list=seen_obj_idx_list_list[chosen_idx],frame_idx_list[chosen_idx],object_description_list_list[chosen_idx],action_description_list_list[chosen_idx],action_type_list_list[chosen_idx]
                    assert len(chosen_seen_obj_idx_list)==1 and len(chosen_object_description_list)==1 and len(chosen_action_description_list)==1 and len(chosen_action_type_list)==1
                    camera_pose_description_list=get_camera_pose_description_v2(annotation_data,chosen_seen_obj_idx_list,chosen_time_idx)
                    obj_no_adj_sentence_list,obj_adj_sentence_list,obj_sentence_list,cam_sentence_list=self.get_seen_objs_description(scene_type,background_description,chosen_object_description_list,chosen_action_description_list,chosen_action_type_list,camera_pose_description_list)
            
                    obj_sentence_list_list.append(obj_sentence_list)
                    obj_adj_sentence_list_list.append(obj_adj_sentence_list)
                    obj_no_adj_sentence_list_list.append(obj_no_adj_sentence_list)
                    cam_sentence_list_list.append(cam_sentence_list)

                    if inital_camera_type_list is None:
                        initial_camera_type_list=camera_pose_description_list
                        
                    
                    if prev_action_type is not None:
                        if prev_action_type!=chosen_action_type_list[0]:
                            action_type_change_list.append(chosen_action_type_list[0])
                            action_type_change_t_idx_list.append(t_idx)
                            
                            
                        if prev_camera_type!=camera_pose_description_list[0]:
                            camera_type_change_list.append(camera_pose_description_list[0])
                            camera_type_change_t_idx_list.append(t_idx)
                            
                    prev_action_type=chosen_action_type_list[0]
                    prev_camera_type=camera_pose_description_list[0]
                    
                
                use_action_change=random.choice([True,True])
                obj_description=obj_sentence_list_list[0][0]
                
                use_back=random.choice([True,False])

                if use_back:
                    ass_template_list=UnrealTrajVideoDataset.BACK_ASSEMBLE_SINGLE_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    obj_description=ass_template.format(background=background_description,first_sentence=obj_description)
                else:
                    ass_template_list=UnrealTrajVideoDataset.NO_BACK_ASSEMBLE_SINGLE_TEMPLATE
                    ass_template=random.choice(ass_template_list)
                    obj_description=ass_template.format(first_sentence=obj_description)
                    
                if use_action_change:    
                    for action_change_t_idx in action_type_change_t_idx_list:
                        action_change_description=obj_no_adj_sentence_list_list[action_change_t_idx][0]
                        then_template=random.choice(UnrealTrajVideoDataset.THEN_TEMPLATE) 
                        obj_description=then_template.format(sentence_1=obj_description,sentence_2=action_change_description)
                
                
                total_description=obj_description

            
            use_descriptor_num=random.uniform(0,1)
            if use_descriptor_num<0.9:
                use_descriptor=True
            else:
                use_descriptor=False
            
            if use_descriptor:
                descriptor_template=random.choice(UnrealTrajVideoDataset.DESCRIPTOR_TEMPLATE)
                total_description=descriptor_template.format(sentence=total_description)      

        
        else:
            total_description=""      
        # print(total_description,img_path_list[frame_indice])
        
        use_descriptor_num=random.uniform(0,1)
        if use_descriptor_num<0.9:
            use_descriptor=True
        else:
            use_descriptor=False
        
        if use_descriptor:
            descriptor_template=random.choice(UnrealTrajVideoDataset.DESCRIPTOR_TEMPLATE)
            background_description=descriptor_template.format(sentence=background_description)      
        
        return total_description,background_description,obj_sentence_list_list,obj_adj_sentence_list_list,objs_info_np_list,total_mask_list,obj_mask_list_list,object_description_list_list



    @classmethod
    def get_enter_and_exit_obj_idx_list(cls,prev_seen_obj_idx_list,seen_obj_idx_list):
        enter_obj_idx_list=[]
        exit_obj_idx_list=[]
        
        enter_obj_idx_list = [idx for idx in seen_obj_idx_list if idx not in prev_seen_obj_idx_list]
        exit_obj_idx_list = [idx for idx in prev_seen_obj_idx_list if idx not in seen_obj_idx_list]
        
        return enter_obj_idx_list, exit_obj_idx_list

    @classmethod
    def sample_clip_from_image_folder(cls,ori_folder, ori_fps, time_duration, clip_time_list,start_frame=None, sample_num=16):
        candidate_list=[]
        
        tgt_fps_min_list=[]
        
        for time_list in clip_time_list:
            start,end=time_list
            video_length=end-start
            # interval = round(ori_fps / tgt_fps)
            if video_length<sample_num:
                continue
            max_interval = math.floor((video_length- 1)/(sample_num - 1))
            
            assert max_interval>0
            candidate_list.append(time_list)
            tgt_fps_min=math.ceil(ori_fps/max_interval)
            tgt_fps_min_list.append(tgt_fps_min)

            
        if not candidate_list:
            return None,None,False
        
        chosen_idx=random.randint(0,len(tgt_fps_min_list)-1)
        candidate=candidate_list[chosen_idx]
        start,end=candidate
        video_length = end-start
        
        
        tgt_fps_min=tgt_fps_min_list[chosen_idx]    
        tgt_fps=random.randint(tgt_fps_min,ori_fps)
        
        
        
        interval = round(ori_fps / tgt_fps)
        if (video_length - (sample_num - 1) * interval - 1<0):
            interval = math.floor(ori_fps / tgt_fps)
            

            
        frame_path_list = [os.path.join(ori_folder, i) for i in os.listdir(ori_folder)]
        frame_path_list = [i for i in frame_path_list if os.path.isfile(i)]
        frame_path_list=sorted(frame_path_list)[:-1]
        
        final_frame_path_list = []
        for frame_path in frame_path_list:
            seq_id, frame_num = os.path.basename(frame_path).split("_")
            frame_num = frame_num.split(".")[0]
            frame_num = int(frame_num)
            if frame_num < start or frame_num >= end:
                continue
            else:
                final_frame_path_list.append(frame_path)

        final_frame_path_list = sorted(final_frame_path_list)
        assert len(final_frame_path_list)==video_length
        if start_frame is None:
            max_start_frame = video_length - (sample_num - 1) * interval - 1
            assert max_start_frame>=0
            start_frame = random.randint(0, max_start_frame)

        frame_indices = [start_frame + i * interval for i in range(sample_num)]

        images = []
        img_path_list=[]
        for idx in frame_indices:
            frame_path = final_frame_path_list[idx]
            # if os.path.exists(frame_path):
            img = imageio.imread(frame_path)
            images.append(img)
            img_path_list.append(frame_path)
        frame_indices=[int(os.path.basename(img_path_list[i]).split("_")[1].split(".")[0]) for i in range(len(img_path_list))]
        return tgt_fps,img_path_list,frame_indices, images,True

    @classmethod
    def sample_video_from_image_folder(cls,ori_folder, ori_fps, time_duration, tgt_fps, start_frame=None, sample_num=16):
        interval = round(ori_fps / tgt_fps)

        frame_path_list = [os.path.join(ori_folder, i) for i in os.listdir(ori_folder)]
        frame_path_list = [i for i in frame_path_list if os.path.isfile(i)]
        frame_path_list=sorted(frame_path_list)[:-1]
        
        final_frame_path_list = []
        video_length = ori_fps * time_duration
        for frame_path in frame_path_list:
            seq_id, frame_num = os.path.basename(frame_path).split("_")
            frame_num = frame_num.split(".")[0]
            frame_num = int(frame_num)
            if frame_num < 0 or frame_num >= video_length:
                continue
            else:
                final_frame_path_list.append(frame_path)

        final_frame_path_list = sorted(final_frame_path_list)
        assert len(final_frame_path_list)==video_length
        if start_frame is None:
            max_start_frame = video_length - (sample_num - 1) * interval - 1
            start_frame = random.randint(0, max_start_frame)

        frame_indices = [start_frame + i * interval for i in range(sample_num)]

        images = []
        img_path_list=[]
        for idx in frame_indices:
            frame_path = final_frame_path_list[idx]
            # if os.path.exists(frame_path):
            img = imageio.imread(frame_path)
            images.append(img)
            img_path_list.append(frame_path)

        return img_path_list,frame_indices, images

    
    def get_clip_time_list(self,idx):
        data_type=self.data_type_list[idx]

        seq_id=self.seq_id_list[idx]
        seq_meta_data=self.seq_meta_data_map[data_type][seq_id]
        
        
        clip_time_list=[]
        
        comment_dict=csv_param_to_dict(seq_meta_data["camera"]["Comment"], str)
        tgt_obj_id_list=eval(comment_dict["tgt_obj_id_list"])
        
        cam_time_range_list=eval(seq_meta_data["camera"]["Time_Range_List"])
        
        prev_tgt_id=None
        for time_range,tgt_id in zip(cam_time_range_list,tgt_obj_id_list):
            if prev_tgt_id is None or tgt_id !=prev_tgt_id:
                clip_time_list.append(time_range)
            else:
                assert clip_time_list[-1][-1]==time_range[0]
                clip_time_list[-1][-1]=time_range[-1]
                
            prev_tgt_id=tgt_id
        
        return clip_time_list
        
    def get_batch(self, idx):
        video_dict = self.dataset[idx]

        video_path=video_dict['clip_path']

        if self.allow_change_tgt:
            tgt_fps=random.choice(self.tgt_fps_list)
            img_path_list,frame_list, images = self.sample_video_from_image_folder(
                video_path, self.ori_fps, self.time_duration, tgt_fps,  sample_num=16
            )
        else:
            clip_time_list=self.get_clip_time_list(idx)
            tgt_fps,img_path_list,frame_list, images ,found= self.sample_clip_from_image_folder(
                video_path, self.ori_fps, self.time_duration,clip_time_list,sample_num=16
            )
            if not found:
                return "",None, "",None,None,None,None,None
            
            
        camera_info_np,intrinsics=self.get_camera_info_np(self.dataset[idx],frame_list)
        caption,background_description,obj_sentence_list_list,obj_adj_sentence_list_list,objs_info_np_list,total_mask_list,obj_mask_list_list,object_description_list_list=self.get_text_prompt_and_mask_list(idx,frame_list)
        
        pixel_values=[]
        for img_path in img_path_list:
            image=Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pixel_value = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
            pixel_values.append(pixel_value / 255.)
        
        pixel_values=torch.stack(pixel_values,axis=0)
        
        
        new_total_mask_list=[]
        
        for total_mask in total_mask_list:
            total_mask=torch.from_numpy(total_mask).permute(2, 0, 1).contiguous()
            new_total_mask_list.append(total_mask)
        
        total_mask=torch.stack(new_total_mask_list,axis=0)
        
        # sphere
        circle_mask_list_list=[]
        
        
        if self.use_sphere_mask:
            # temp_save_dir="circle_mask"
            for time_idx,obj_mask_list in enumerate(obj_mask_list_list):
                circle_mask_list=[]
                for mask_idx,mask in enumerate(obj_mask_list):
                    y, x = np.nonzero(mask[...,0])
                    gaussian_mask=mask[...,0]
                    if len(x) > 0 and len(y) > 0:
                        center, radius = cv2.minEnclosingCircle((np.column_stack((x, y))).astype(np.float32))
                        circle_mask = np.zeros_like(mask).astype(np.uint8)
                        # cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, 0)
                        # Draw a filled circle instead of just the outline
                        cv2.circle(circle_mask, (int(center[0]), int(center[1])), int(radius), 1, -1)
                        circle_mask=(circle_mask>0)
                        # Generate a Gaussian circle mask
                        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
                        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                        
                        # Create a Gaussian distribution
                        # sigma = radius / 3  # Adjust this value to control the softness of the edge
                        sigma = radius / 2  # Adjust this value to control the softness of the edge
                        # sigma = 10
                        # sigma = radius**2
                        gaussian_mask = np.exp(-0.5 * (dist_from_center / sigma)**2)
                        
                        
                        # Normalize the Gaussian mask
                        gaussian_mask = gaussian_mask / gaussian_mask.max()

                        
                        gaussian_mask=circle_mask[...,0]*gaussian_mask

                        # _circle_mask=gaussian_mask
                        
                        # img = Image.fromarray((_circle_mask*255).astype(np.uint8))

                        # img.save(os.path.join(temp_save_dir,f"{time_idx}-{mask_idx}.png"))
                        
                        # ord_mask=mask[...,0]
                        
                        # img = Image.fromarray((ord_mask).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"mask-{time_idx}-{mask_idx}.png"))
                        
                        
                        # img = Image.fromarray((circle_mask[...,0]).astype(np.uint8)*255)

                        # img.save(os.path.join(temp_save_dir,f"circle-mask-{time_idx}-{mask_idx}.png"))
                        # pass
                    circle_mask_list.append(torch.from_numpy(gaussian_mask[...,None]).permute(2, 0, 1).contiguous())
                
                # circle_mask_list_list.append(circle_mask_list)
                if circle_mask_list:
                    circle_mask_list_list.append(torch.stack(circle_mask_list))
                    

        circle_mask_list=circle_mask_list_list
        
        new_obj_mask_list_list=[]
        for obj_mask_list in obj_mask_list_list:
            new_obj_mask_list=[]
            for obj_mask in obj_mask_list:
                new_obj_mask_list.append(torch.from_numpy(obj_mask).permute(2, 0, 1).contiguous())
            if new_obj_mask_list:
                new_obj_mask_list_list.append(torch.stack(new_obj_mask_list))
            else:
                assert caption==""
            
        obj_mask_list=new_obj_mask_list_list


        objs_info_list=[]
        for obj_info in objs_info_np_list:
            objs_info_list.append(torch.from_numpy(obj_info))
        
        return video_path,pixel_values, caption,background_description,torch.from_numpy(camera_info_np),objs_info_list,total_mask,obj_mask_list,frame_list,torch.from_numpy(intrinsics),circle_mask_list,obj_sentence_list_list,obj_adj_sentence_list_list,object_description_list_list

    @classmethod
    def get_camera_info_np(cls,label_data,frame_idx_list):
        annotation_file_path=label_data["annotation_file_path"]
        with open(annotation_file_path,"r") as f:
            annotation_data=json.load(f)
        cam_dict=annotation_data["camera"]
    
        cam_info_list=[]
        
        intrinsics=[]
        for time_idx in frame_idx_list:
            cam_xyz,cam_euler_rot=cam_dict[time_idx][:3],cam_dict[time_idx][3:6]
            cam_rot=transform_euler_to_matrix_v2(cam_euler_rot[2],cam_euler_rot[1],cam_euler_rot[0]) ##,roll, pitch , yaw
            
            cam_rot=np.array(cam_rot)

            cam_info=np.eye(4)
            
            cam_info[:3,:3]=cam_rot
            cam_info[:3,3]=np.array(cam_xyz)
 
            cam_info_list.append(cam_info)
            
            
            intrinsic=cam_dict[time_idx][-3:-1]+[0,0]
            intrinsics.append(intrinsic)
            
        
        cam_info_np=np.stack(cam_info_list)
        intrinsics=np.stack(intrinsics)
        return cam_info_np,intrinsics
            
            
    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        while True:
            video_path,video, video_caption,background_description,camera_info,obj_info_list,total_mask,obj_mask_list,frame_list,intrinsics,circle_mask_list,obj_sentence_list_list,obj_adj_sentence_list_list,object_description_list_list = self.get_batch(idx)
            if video_caption!="":
                break
            else:
                idx = random.randint(0, self.length - 1)
                continue

        video = self.pixel_transforms(video)

        video = rearrange(video, "t c h w-> c t h w") ## t c h w-> c t h w

        total_mask=self.mask_transforms(total_mask)
        
        new_obj_mask_list=[]
        for obj_mask in obj_mask_list:
            new_obj_mask_list.append(self.mask_transforms(obj_mask))
            
        obj_mask_list=new_obj_mask_list
        
        
        
        new_circle_mask_list=[]
        for circle_mask in circle_mask_list:
            new_circle_mask_list.append(self.mask_transforms(circle_mask))

        circle_mask_list=new_circle_mask_list
        
        ord_camera_info=camera_info
        camera_info,obj_info_list=self.get_ref_matrix(camera_info,obj_info_list,self.cam_translation_rescale_factor,self.obj_translation_rescale_factor)

        first_frame_camera = ord_camera_info[0]
        rotation_matrix = first_frame_camera[:3,:3]
        zero_translation = torch.zeros(3)
        new_first_camera_matrix = torch.zeros((3,4))
        new_first_camera_matrix[:3, :3] = rotation_matrix
        new_first_camera_matrix[:3, 3] = zero_translation
        
        # Flatten the new camera matrix to a 12-element vector
        new_first_camera_info = new_first_camera_matrix.reshape(-1)
        
        # Repeat this new camera info for all frames
        camera_info[0]=new_first_camera_info
        
        sample = dict(video_path=video_path,pixel_values=video, caption=video_caption,background_description=background_description,camera_info=camera_info,obj_info_list=obj_info_list,obj_mask_list=obj_mask_list,frame_list=frame_list,intrinsics=intrinsics,circle_mask_list=circle_mask_list,obj_sentence_list=obj_sentence_list_list,obj_adj_sentence_list=obj_adj_sentence_list_list,object_description_list=object_description_list_list)
        # sample = dict(video_path=video_path,pixel_values=video, caption=video_caption,camera_info=camera_info,obj_info_list=obj_info_list)

        return sample

    
    @classmethod
    def get_ref_matrix(cls,camera_info,obj_info_list,cam_translation_rescale_factor,obj_translation_rescale_factor):
        ##get camera to obj
        
        new_camera_info=create_relative_matrix_of_cam_list(camera_info,cam_translation_rescale_factor)
        
        new_obj_info_list=[]
        
        for cam_info,obj_info in zip(camera_info,obj_info_list):
            new_obj_info_list.append(create_relative_matrix_of_two_torch_matrix(cam_info,obj_info,scale_T = obj_translation_rescale_factor))
        
        return new_camera_info,new_obj_info_list
        ##get camera to camera
        
        
    
    @classmethod
    def collate_fn(cls,batch):
        # Collate function for DataLoader
        video_paths = [item['video_path'] for item in batch]
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        captions = [item['caption'] for item in batch]
        background_captions = [item['background_description'] for item in batch]
        camera_infos = torch.stack([item['camera_info'] for item in batch])
        obj_info_list_list = [item['obj_info_list'] for item in batch]
        obj_mask_list_list= [item['obj_mask_list'] for item in batch]
        frame_list_list=[item['frame_list'] for item in batch]
        
        intrinsics=torch.stack([item['intrinsics'] for item in batch])
        
        circle_mask_list_list=[item['circle_mask_list'] for item in batch]
        
        obj_sentence_list_list=[item['obj_sentence_list'] for item in batch]
        
        obj_adj_sentence_list_list=[item['obj_adj_sentence_list'] for item in batch]
        
        object_description_list_list=[item['object_description_list'] for item in batch]


        return {
            'video_paths': video_paths,
            'videos': pixel_values,
            'captions': captions,
            'background_captions': background_captions,
            'camera_infos': camera_infos,
            'obj_info_list_list': obj_info_list_list,
            'obj_mask_list_list': obj_mask_list_list,
            'frame_list_list':frame_list_list,
            "intrinsics":intrinsics,
            "circle_mask_list_list":circle_mask_list_list,
            "obj_sentence_list_list":obj_sentence_list_list,
            "obj_adj_sentence_list_list":obj_adj_sentence_list_list,
            "object_description_list_list":object_description_list_list,
        }
