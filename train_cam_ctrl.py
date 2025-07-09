import os
import math
import random
import time
import inspect
import argparse
import datetime
import subprocess
import json
import csv
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple
from PIL import Image


import numpy as np
from einops import rearrange
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer


from fmc.utils.util import setup_logger, format_time, save_videos_grid
from fmc.pipelines.pipeline_animation import CameraCtrlPipeline
from fmc.models.unet import UNet3DConditionModelPoseCond
from fmc.models.pose_adaptor import CameraPoseEncoder, PoseAdaptor
from fmc.models.attention_processor import AttnProcessor as CustomizedAttnProcessor
from fmc.data.dataset import UnrealTrajVideoDataset,UnrealTrajLoraDataset,ray_condition
from fmc.data.utils import create_absolute_matrix_from_ref_cam_list


def save_camera_info_to_txt_file(save_root,abs_camera_info_list,camera_info_batch,obj_info_batch,prompts,img_path_list_list,cam_translation_rescale_factor,obj_translation_rescale_factor):
    for idx in range(len(camera_info_batch)):
        prompt=prompts[idx]
        with open(os.path.join(save_root, f"label_{idx}.txt"), "w") as f:
            f.write(prompt+"\n")
            for img_path in img_path_list_list[idx]:
                f.write(img_path+"\n")
        
        
        frame_num,dim=camera_info_batch[idx].shape
        ref_cam_info_t=camera_info_batch[idx].reshape(frame_num,3,4)
        with open(os.path.join(save_root, f"cam_label_compute_{idx}.txt"), "w") as f:
            absolute_cam_info_list=create_absolute_matrix_from_ref_cam_list(abs_camera_info_list[idx][0].cpu().numpy(),ref_cam_info_t.cpu().numpy(),scale_T = cam_translation_rescale_factor)
            f.write("labels \n")
            for cam_info in absolute_cam_info_list:
                f.write("-1 -1 -1 -1 -1 -1 -1 ")
                cam_info=cam_info.reshape(12)
                cam=[]
                for i in cam_info:
                    cam.append(str(i))
                cam_str=" ".join(cam)
                f.write(f"{cam_str}\n")
        
        with open(os.path.join(save_root, f"cam_label_gt_{idx}.txt"), "w") as f:
            f.write("gt labels \n")
            for cam_info in abs_camera_info_list[idx]:
                f.write("-1 -1 -1 -1 -1 -1 -1 ")
                cam_info=cam_info[:3,:].reshape(12)
                cam=[]
                for i in cam_info:
                    cam.append(str(i))
                cam_str=" ".join(cam)
                f.write(f"{cam_str}\n")
                    
            
            
def to_plucker_embedding(c2w_rel_poses,intrinsics,sample_size,ori_h=None, ori_w=None,rescale_fxy=True):
    intrinsics = torch.as_tensor(intrinsics)                 
    c2w = torch.as_tensor(c2w_rel_poses)                     
    B, n_frame, _, _ = c2w.shape
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=c2w.dtype, device=c2w.device)
    bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, n_frame, 1, 4)
    c2w = torch.cat([c2w, bottom_row], dim=2)

    flip_flag = torch.zeros(16, dtype=torch.bool, device=c2w.device)

    plucker_embedding = ray_condition(intrinsics, c2w, sample_size[0], sample_size[1], device='cpu',
                                        flip_flag=flip_flag).permute(0,1, 4, 2, 3).contiguous()
    
    return plucker_embedding

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
    elif launcher=="single":
        local_rank=0
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

    return local_rank


def main(name: str,
         launcher: str,
         port: int,

         output_dir: str,
         pretrained_model_path: str,

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         unet_additional_kwargs: Dict = {},
         unet_subfolder: str = "unet",

         lora_rank: int = 4,
         lora_scale: float = 1.0,
         lora_ckpt: str = None,
         motion_module_ckpt: str = "",
         motion_lora_rank: int = 0,
         motion_lora_scale: float = 1.0,

         pose_encoder_kwargs: Dict = None,
         attention_processor_kwargs: Dict = None,
         noise_scheduler_kwargs: Dict = None,

         do_sanity_check: bool = True,

         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,

         global_seed: int = 42,
         logger_interval: int = 10,
         resume_from: str = None,
         
         apply_masked_loss=False,
         mask_loss_weight=0,
         sd_loss_weight=1,
         
         appearance_debias=0,
         
         is_debug=False,
         train_unet=False,
         train_mm=False,
         ):
    check_min_version("0.10.0.dev0")

    local_rank = init_dist(launcher=launcher, port=port)

    if launcher!="single":
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
    else:
        global_rank = 0
        num_processes = 1
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    logger = setup_logger(output_dir, global_rank)

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(pretrained_model_path, subfolder=unet_subfolder,
                                                           unet_additional_kwargs=unet_additional_kwargs)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)

    logger.info(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=lora_ckpt is not None,
                                add_motion_lora=motion_lora_rank > 0,
                                lora_kwargs={"lora_rank": lora_rank, "lora_scale": lora_scale},
                                motion_lora_kwargs={"lora_rank": motion_lora_rank, "lora_scale": motion_lora_scale},
                                **attention_processor_kwargs)

    if lora_ckpt is not None:
        logger.info(f"Loading the image lora checkpoint from {lora_ckpt}")
        lora_checkpoints = torch.load(lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        logger.info(f'Loading done')
    else:
        logger.info(f'We do not add image lora')


    if motion_module_ckpt != "":
        logger.info(f"Loading the motion module checkpoint from {motion_module_ckpt}")
        mm_checkpoints = torch.load(motion_module_ckpt, map_location=unet.device)
        if 'motion_module_state_dict' in mm_checkpoints:
            mm_checkpoints = {k.replace('module.', ''): v for k, v in mm_checkpoints['motion_module_state_dict'].items()}
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        logger.info("Loading done")
    else:
        logger.info(f"We do not load pretrained motion module checkpoint")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    spatial_attn_proc_modules = torch.nn.ModuleList([v for v in unet.attn_processors.values()
                                                     if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    temporal_attn_proc_modules = torch.nn.ModuleList([v for v in unet.mm_attn_processors.values()
                                                      if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    spatial_attn_proc_modules.requires_grad_(True)
    temporal_attn_proc_modules.requires_grad_(True)
    pose_encoder.requires_grad_(True)

    for n, p in spatial_attn_proc_modules.named_parameters():
        if 'lora' in n:
            p.requires_grad = False
            logger.info(f'Setting the `requires_grad` of parameter {n} to false')
    pose_adaptor = PoseAdaptor(unet, pose_encoder)

    encoder_trainable_params = list(filter(lambda p: p.requires_grad, pose_encoder.parameters()))
    encoder_trainable_param_names = [p[0] for p in
                                     filter(lambda p: p[1].requires_grad, pose_encoder.named_parameters())]
    attention_trainable_params = [v for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]
    attention_trainable_param_names = [k for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]

    trainable_params = encoder_trainable_params + attention_trainable_params
    trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names
    

    mm_params=[]
    mm_param_names=[]
    if train_mm:
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "TemporalTransformer3DModel":
                mm_param_names.append(f'{_name}.norm')
                mm_param_names.append(f'{_name}.proj_in')            
                mm_param_names.append(f'{_name}.proj_out')


    for __name, param in unet.named_parameters():
        for _trainable_module_name in mm_param_names:
            if _trainable_module_name in __name:
                param.requires_grad = True
                mm_params.append(param)
                break
    
    trainable_params+=mm_params
    trainable_param_names+=mm_param_names
                      

    if is_main_process:
        logger.info(f"trainable parameter number: {len(trainable_params)}")
        logger.info(f"encoder trainable number: {len(encoder_trainable_params)}")
        logger.info(f"attention processor trainable number: {len(attention_trainable_params)}")
        logger.info(f"mm trainable number: {len(mm_params)}")
        
        logger.info(f"trainable parameter names: {trainable_param_names}")
        logger.info(f"encoder trainable scale: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
        logger.info(f"attention processor trainable scale: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
        logger.info(f"mm scale: {sum(p.numel() for p in mm_params) / 1e6:.3f} M")
        
        logger.info(f"trainable parameter scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    vae.to(local_rank)
    text_encoder.to(local_rank)


    logger.info(f'Building training datasets')
    train_dataset = UnrealTrajVideoDataset(**train_data.params)
    
    if launcher!="single":
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            rank=global_rank,
            shuffle=True,
            seed=global_seed,
            
        )


        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=distributed_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=UnrealTrajVideoDataset.collate_fn,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=UnrealTrajVideoDataset.collate_fn,
        )   


    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)


    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if is_main_process:
        validation_pipeline = CameraCtrlPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            pose_encoder=pose_encoder)
        validation_pipeline.enable_vae_slicing()

        with open(validation_data.get("hdri_json_file_path"),"r") as f:
            hdri_json_data=json.load(f)
        
        with open(validation_data.get("asset_json_file_path"),"r") as f:
            asset_json_data=json.load(f)
        
        seq_meta_data_map={}##(["static_single","static_multi","dynamic_single","dynamic_multi"])
        seq_id_max_map=validation_data.get("seq_id_max_map")
        
        for dynamic_type_str in ["static","dynamic"]:
            for single_type_str in ["","_multi"]:
                seq_meta_data={}
                csv_path=os.path.join(validation_data["seq_csv_root"],f"traj_{dynamic_type_str}{single_type_str}.csv")
                with open(csv_path, mode="r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    csv_rows = list(
                        csv_reader
                    )  # Convert to list of rows so that we can look ahead, this will skip header

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
                
    
    

    pose_adaptor.to(local_rank)
    if launcher!="single":
        pose_adaptor = DDP(pose_adaptor, device_ids=[local_rank], output_device=local_rank)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    


    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu")


        
        global_step = ckpt['global_step']
        trained_iterations = (global_step % len(train_dataloader))
        first_epoch = int(global_step // len(train_dataloader))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        pose_encoder_state_dict = ckpt['pose_encoder_state_dict']
        attention_processor_state_dict = ckpt['attention_processor_state_dict']
        if launcher!="single":
            pose_enc_m, pose_enc_u = pose_adaptor.module.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        else:
            pose_enc_m, pose_enc_u = pose_adaptor.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)

        assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
        if launcher!="single":
            _, attention_processor_u = pose_adaptor.module.unet.load_state_dict(attention_processor_state_dict, strict=False)
        else:
            _, attention_processor_u = pose_adaptor.unet.load_state_dict(attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        logger.info(f"Loading the pose encoder and attention processor weights done.")
        logger.info(f"Loading done, resuming training from the {global_step + 1}th iteration")
        lr_scheduler.last_epoch = first_epoch
        
    else:
        trained_iterations = 0

    
    ord_global_step=global_step
    ord_first_epoch=first_epoch

    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        if launcher!="single":
            train_dataloader.sampler.set_epoch(epoch)
        pose_adaptor.train()

        data_iter = iter(train_dataloader)
        for step in range(trained_iterations, len(train_dataloader)):

            iter_start_time = time.time()
            batch = next(data_iter)
            data_end_time = time.time()
            
            back_num=random.uniform(0,1)
            if back_num<0.4:
                use_back=True
            else:
                use_back=False

            if use_back:
                batch['text']=batch["background_captions"]
            else:
                batch['text']=batch["captions"]
            
            batch["pixel_values"]=rearrange(batch["videos"], "b c f h w -> b f c h w")
            
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]

            if epoch == first_epoch and step == 0 and do_sanity_check:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif",
                                     rescale=True)

            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

            noise = torch.randn_like(latents)  # [b, c, f, h, w]
            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # [b, c, f h, w]

            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]  # b l c

            enable_cmcm=True
            is_cm_condition_null_list=[]
            if enable_cmcm:
                RT = batch["camera_infos"].to(device=local_rank,dtype=pixel_values.dtype)
                RT = RT[...] # [b, t, 12]
                RT=RT.reshape(RT.shape[0],RT.shape[1],3,4)
                if cfg_random_null_text:
                    for i in range(RT.shape[0]):
                        is_not_null= random.random() > cfg_random_null_text_ratio
                        RT[i] = RT[i] if is_not_null else torch.zeros_like(RT[i])
                        is_cm_condition_null_list.append(not is_not_null)
                else:
                    is_cm_condition_null_list=[False for _ in range(RT.shape[0])]
                    
            else:
                RT=None
                is_cm_condition_null_list=[True for _ in range(16)]
            
            
            intrinsics=batch["intrinsics"]
            batch["plucker_embedding"]=to_plucker_embedding(RT,intrinsics,train_data.params.sample_size)
            plucker_embedding = batch["plucker_embedding"].to(device=local_rank)  # [b, f, 6, h, w]
            plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")  # [b, 6, f h, w]
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = pose_adaptor(noisy_latents,
                                          timesteps,
                                          encoder_hidden_states=encoder_hidden_states,
                                          pose_embedding=plucker_embedding)  # [b c f h w]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                sd_loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                if apply_masked_loss:
                    obj_info_list_list=batch['obj_info_list_list']
                    obj_mask_list_list=batch['obj_mask_list_list']
                    B=len(obj_info_list_list)
                    F=len(obj_info_list_list[0])
                    
                    H,W=obj_mask_list_list[0][0].shape[-2],obj_mask_list_list[0][0].shape[-1]
                    mask=torch.zeros([B,F,H,W,1],dtype=pixel_values.dtype).to(local_rank)             
                    for b_idx,(obj_info_list, obj_mask_list) in enumerate(zip(obj_info_list_list, obj_mask_list_list)):
                        for f_idx,(obj_info, obj_mask) in enumerate(zip(obj_info_list, obj_mask_list)):
                            obj_mask=obj_mask.permute(0,2,3,1).to(local_rank)
                            for _obj_mask in obj_mask:
                                mask[b_idx][f_idx][_obj_mask[...,0]]=1
                    
                    
                    mask=rearrange(mask,"b f h w c-> (b f) c h w")##B frame c H w
                    ord_h,ord_w=model_pred.shape[-2],model_pred.shape[-1]
                    mask = torch.nn.functional.interpolate(
                        input=mask, size=(ord_h,ord_w)
                    )
                    
                    mask=rearrange(mask,"(b f) c h w -> b c f h w",b=B)##B c frame H w
                    
                    mask=1-mask## revert
                    
                    masked_model_pred = mask *model_pred 
                    masked_target = mask*target
                    mask_loss=torch.nn.functional.mse_loss(masked_model_pred.float(), masked_target.float(), reduction="mean")
                    loss = mask_loss_weight*mask_loss+sd_loss_weight*sd_loss

                else:
                    loss=sd_loss
                
                                
                if appearance_debias > 0: # appearance debias loss from MotionDirector (https://arxiv.org/abs/2310.08465)
                    anchor = []
                    for i in range(target.shape[0]):
                        randidx = random.randint(0, target.shape[2]-1)##B c frame H w
                        anchor.append(target[i:i+1, :, randidx:randidx+1, :, :])
                    anchor = torch.cat(anchor, dim=0)
                    anchor = anchor.repeat_interleave(target.shape[2], dim=2)
                    loss_app_debias = torch.nn.functional.mse_loss(math.sqrt(2) * model_pred - anchor,
                                                math.sqrt(2) * target - anchor, 
                                                reduction="mean")
                    loss = loss + appearance_debias * loss_app_debias

            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pose_adaptor.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pose_adaptor.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            iter_end_time = time.time()

            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "pose_encoder_state_dict": pose_adaptor.module.pose_encoder.state_dict() if launcher!="single" else pose_adaptor.pose_encoder.state_dict(),
                    "attention_processor_state_dict": {k: v for k, v in unet.state_dict().items()
                                                       if k in attention_trainable_param_names},
                    "optimizer_state_dict": optimizer.state_dict()
                }
                if train_mm:
                    state_dict["mm_state_dict"]={k: v for k, v in unet.state_dict().items()
                                                       if k in mm_param_names}
                    
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            if is_main_process and ((global_step ==1+ord_global_step) or (
                    (global_step + 1) % validation_steps == 0 or (global_step + 1) in validation_steps_tuple)):

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                random.seed(global_seed)
                
                height = validation_data.sample_size[0] if not isinstance(validation_data.sample_size, int) else validation_data.sample_size
                width  = validation_data.sample_size[1] if not isinstance(validation_data.sample_size, int) else validation_data.sample_size

                
                mask_transforms = [transforms.Resize(validation_data.sample_size)]
        
                mask_transforms = transforms.Compose(mask_transforms)
                kwargs={
                    "data_root":validation_data.get("data_root"),
                    "label_root":validation_data.get("label_root"),
                    "mask_root":validation_data.get("mask_root"),
                    "seq_meta_data_map":seq_meta_data_map,##(["static_single","static_multi","dynamic_single","dynamic_multi"])
                    "seq_id_max_map":seq_id_max_map,##(["static_single","static_multi","dynamic_single","dynamic_multi"])
                    "hdri_json_data":hdri_json_data,
                    "asset_json_data":asset_json_data,
                    "cam_translation_rescale_factor":validation_data.get("cam_translation_rescale_factor"),
                    "obj_translation_rescale_factor":validation_data.get("obj_translation_rescale_factor"),
                    "allow_change_tgt":validation_data.get("allow_change_tgt"),
                    "tgt_fps_list":validation_data.get("tgt_fps_list"),
                    "ori_fps":validation_data.get("ori_fps"),
                    "time_duration":validation_data.get("time_duration") ,
                    "mask_transforms":mask_transforms,
                    "use_sphere_mask":train_data.params.use_sphere_mask,
                    "height":validation_data.sample_size[0],
                    "width":validation_data.sample_size[1],                                 
                }
                
    
                prompts=[]        
                batch_cam_info_list=[]  
                abs_camera_info_list=[]
                batch_obj_info_list=[]      
                img_path_list_list=[]
                for i in range(validation_data.num):
                    generator = torch.Generator(device=local_rank)
                    generator.manual_seed(global_seed)
                    prompt,intrinsics,abs_camera_info,camera_info,_,objs_info_list,obj_mask_list,frame_idx_list,img_path_list,circle_mask_list_list,_=train_dataset.__class__.create_validation_prompts(**kwargs)                    
                    if validation_data.change_obj_back:
                        obj_prompts=UnrealTrajLoraDataset.create_validation_prompts_without_cam(1,use_synthetic_des=True,max_obj_num=1)                  
                        ord_prompt=prompt
                        prompt=obj_prompts[0]
                        prompt.replace("image","video")
                    
                    real_images = []
                    prompts.append(prompt)
                    img_path_list_list.append(img_path_list)
                    for _, img_path in enumerate(img_path_list):
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        real_images.append(torchvision.transforms.functional.to_tensor(img))
                        
                    real_save_path = f"{output_dir}/samples/{global_step}/real-{i}.gif"
                    real_images=torch.stack(real_images) ##t c h w
                    real_images=real_images.permute(1,0,2,3).unsqueeze(0)##b c t h w
                    save_videos_grid(real_images, real_save_path)
                    
                    logger.info(f"Saved real images to {real_save_path}. prompt is {prompt}")
                    
                    
                    RT = camera_info.unsqueeze(0).to(device=local_rank,dtype=pixel_values.dtype)
                    
                    batch_cam_info_list.append(RT.squeeze(0))
                    abs_camera_info_list.append(abs_camera_info.to(device=local_rank,dtype=pixel_values.dtype))

                    RT=RT.reshape(RT.shape[0],RT.shape[1],3,4)
                    intrinsics=intrinsics.unsqueeze(0)
                    plucker_embedding=to_plucker_embedding(RT,intrinsics,validation_data.sample_size).to(device=local_rank)
                    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")  # [b, 6, f h, w]
                    
                    sample = validation_pipeline(
                        prompt=prompt,
                        pose_embedding=plucker_embedding,
                        video_length=video_length,
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.,
                        generator=generator,
                    ).videos[0]  # [3 f h w]
                    save_path = f"{output_dir}/samples/{global_step}/cm-{i}.gif"
                    save_videos_grid(sample[None, ...], save_path)
                
                prompt_save_root=save_path = f"{output_dir}/samples/{global_step}"
            
                save_camera_info_to_txt_file(prompt_save_root,abs_camera_info_list,batch_cam_info_list,batch_obj_info_list,prompts,img_path_list_list,validation_data.get("cam_translation_rescale_factor"),validation_data.get("obj_translation_rescale_factor"))

            if (global_step % logger_interval) == 0 or global_step == 0:
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg = f"Iter: {global_step}/{max_train_steps}, Loss: {loss.detach().item(): .4f}, " \
                      f"lr: {lr_scheduler.get_last_lr()}, Data time: {format_time(data_end_time - iter_start_time)}, " \
                      f"Iter time: {format_time(iter_end_time - data_end_time)}, " \
                      f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}, " \
                      f"GPU memory: {gpu_memory: .2f} G"
                logger.info(msg)

            if global_step >= max_train_steps:
                break
            
    if launcher!="single":
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cam.yaml")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm","single"], default="single")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, **config)