#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from kornia import create_meshgrid
import torch

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def set_rays_od(cams):
    for id, cam in enumerate(cams):
        rayd=1
        if rayd is not None:
            projectinverse = cam.projection_matrix.T.inverse()
            camera2wold = cam.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(cam.image_height, cam.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            xindx = pixgrid[:,:,0] # x
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, cam.image_height), pix2ndc(xindx, cam.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4
            projected = ndccamera @ projectinverse.T
            diretioninlocal = projected / projected[:,:,3:] #v
            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T
            # rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            rays_d = direction
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            cam.rayo = cam.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()
            cam.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()
        else :
            cam.rayo = None
            cam.rayd = None

def set_rays(scene,resolution_scales=[1.0]):
    set_rays_od(scene.getTrainCameras())
    for resolution_scale in resolution_scales:
        for cam in scene.train_cameras[resolution_scale]:
            if cam.rayo is not None:
                cam.rays = torch.cat([cam.rayo, cam.rayd], dim=1)
