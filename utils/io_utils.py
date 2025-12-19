"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import pickle
from pathlib import Path

from third_party.stylegan2_ada_pytorch import dnnlib
from utils import latent_space_ops

import numpy as np
import torch
import torchvision
from PIL import Image

IMAGE_SUFFIX = ['.jpg', '.png', '.svg', '.webp', '.jpeg']


def str2bool(s):
    if s.lower() in ['false', 'f', 'no']:
        return False
    if s.lower() in ['true', 't', 'yes']:
        return True

    raise ValueError(f"Don't know how to convert {s} to bool")


def float_or_none(s):
    if s.lower() == 'none':
        return None
    else:
        return float(s)


def existing_path(s):
    p = Path(s)
    if not p.exists():
        raise ValueError(f'Input path {s} does not exist but is expected to')

    return p


def create_path(s):
    p = Path(s)
    p.mkdir(exist_ok=True, parents=True)
    return p


def load_single_latent(latent_path: Path):
    suffix = latent_path.suffix
    if suffix == '.pt':
        x = torch.load(latent_path)
    elif suffix == '.npy':
        x = torch.FloatTensor(np.load(latent_path))
    elif suffix == '.pickle':
        with open(latent_path, 'rb') as fp:
            x = pickle.load(fp)
    else:
        raise NotImplemented()

    return x.to('cuda')


def load_latents(latents_dir: Path, to_w=False):
    latents = []
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue

        latents.append(torch.load(f))
    latents = torch.stack(latents, dim=0)

    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    return latents


def load_latents_fb_multifaces(latents_dir: Path, to_w=False):
    latents = []
    latents_exp_dict = {}
    latents_frame_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp, frame = image_name_tokens[1], image_name_tokens[-2]
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)
        
        if exp in latents_exp_dict.keys():
            latents_exp_dict[exp].append(latent_pt)
        else:
            latents_exp_dict[exp] = [latent_pt]
        
        if frame in latents_frame_dict.keys():
            latents_frame_dict[frame].append(latent_pt)
        else:
            latents_frame_dict[frame] = [latent_pt]

    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_frame_dict.keys():
        latents_frame_dict[k]= torch.stack(latents_frame_dict[k], dim=0)
        if to_w:
            latents_frame_dict[k] = latent_space_ops.wplus_to_w(latents_frame_dict[k])
    
    return latents, latents_exp_dict, latents_frame_dict


def load_latents_lightstage(latents_dir: Path, to_w=False):
    latents = []
    latents_exp_dict = {}
    latents_frame_dict = {}
    latents_light_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp, frame, light = image_name_tokens[1], image_name_tokens[-2], image_name_tokens[-1].split('.')[0]
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)
        
        if exp in latents_exp_dict.keys():
            latents_exp_dict[exp].append(latent_pt)
        else:
            latents_exp_dict[exp] = [latent_pt]
        
        if frame in latents_frame_dict.keys():
            latents_frame_dict[frame].append(latent_pt)
        else:
            latents_frame_dict[frame] = [latent_pt]
        
        if light in latents_light_dict.keys():
            latents_light_dict[light].append(latent_pt)
        else:
            latents_light_dict[light] = [latent_pt]

    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_frame_dict.keys():
        latents_frame_dict[k]= torch.stack(latents_frame_dict[k], dim=0)
        if to_w:
            latents_frame_dict[k] = latent_space_ops.wplus_to_w(latents_frame_dict[k])

    for k in latents_light_dict.keys():
        latents_light_dict[k]= torch.stack(latents_light_dict[k], dim=0)
        if to_w:
            latents_light_dict[k] = latent_space_ops.wplus_to_w(latents_light_dict[k])
    
    return latents, latents_exp_dict, latents_frame_dict, latents_light_dict


def load_latents_fb_multifaces_coord_access(latents_dir: Path, to_w=False):
    '''
    anchor loss of PCA applied.
    load latents for coordinate access
    '''
    latents = []
    latents_exp_dict = {}
    latents_view_dict = {}
    latents_exp_dict_dict = {}
    latents_view_dict_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp_name, view_name = image_name_tokens[1], image_name_tokens[-2]
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)

        if exp_name in latents_exp_dict.keys():
            latents_exp_dict[exp_name].append(latent_pt)
            latents_exp_dict_dict[exp_name][view_name] = latent_pt
        else:
            latents_exp_dict[exp_name] = [latent_pt]
            latents_exp_dict_dict[exp_name] = {view_name: latent_pt}

        if view_name in latents_view_dict.keys():
            latents_view_dict[view_name].append(latent_pt)
            latents_view_dict_dict[view_name][exp_name] = latent_pt
        else:
            latents_view_dict[view_name] = [latent_pt]
            latents_view_dict_dict[view_name] = {exp_name: latent_pt}
    
    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_view_dict.keys():
        latents_view_dict[k]= torch.stack(latents_view_dict[k], dim=0)
        if to_w:
            latents_view_dict[k] = latent_space_ops.wplus_to_w(latents_view_dict[k])

    return latents, latents_exp_dict, latents_view_dict, latents_exp_dict_dict, latents_view_dict_dict


def load_latents_obama_coord_access(latents_dir: Path, to_w=False):
    '''
    anchor loss of PCA applied.
    load latents for coordinate access
    '''
    latents = []
    latents_exp_dict = {}
    latents_view_dict = {}
    latents_exp_dict_dict = {}
    latents_view_dict_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        import pdb; pdb.set_trace()
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp_name, view_name = image_name_tokens[-2], image_name_tokens[1] + '_' + image_name_tokens[2]
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)

        if exp_name in latents_exp_dict.keys():
            latents_exp_dict[exp_name].append(latent_pt)
            latents_exp_dict_dict[exp_name][view_name] = latent_pt
        else:
            latents_exp_dict[exp_name] = [latent_pt]
            latents_exp_dict_dict[exp_name] = {view_name: latent_pt}

        if view_name in latents_view_dict.keys():
            latents_view_dict[view_name].append(latent_pt)
            latents_view_dict_dict[view_name][exp_name] = latent_pt
        else:
            latents_view_dict[view_name] = [latent_pt]
            latents_view_dict_dict[view_name] = {exp_name: latent_pt}
    
    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_view_dict.keys():
        latents_view_dict[k]= torch.stack(latents_view_dict[k], dim=0)
        if to_w:
            latents_view_dict[k] = latent_space_ops.wplus_to_w(latents_view_dict[k])

    return latents, latents_exp_dict, latents_view_dict, latents_exp_dict_dict, latents_view_dict_dict


def load_latents_celebrities_coord_access(latents_dir: Path, to_w=False):
    '''
    anchor loss of PCA applied.
    load latents for coordinate access
    '''
    latents = []
    latents_exp_dict = {}
    latents_view_dict = {}
    latents_exp_dict_dict = {}
    latents_view_dict_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp_name, view_name = image_name_tokens[-3], image_name_tokens[2] + '_' + image_name_tokens[3]
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)

        if exp_name in latents_exp_dict.keys():
            latents_exp_dict[exp_name].append(latent_pt)
            latents_exp_dict_dict[exp_name][view_name] = latent_pt
        else:
            latents_exp_dict[exp_name] = [latent_pt]
            latents_exp_dict_dict[exp_name] = {view_name: latent_pt}

        if view_name in latents_view_dict.keys():
            latents_view_dict[view_name].append(latent_pt)
            latents_view_dict_dict[view_name][exp_name] = latent_pt
        else:
            latents_view_dict[view_name] = [latent_pt]
            latents_view_dict_dict[view_name] = {exp_name: latent_pt}
    
    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_view_dict.keys():
        latents_view_dict[k]= torch.stack(latents_view_dict[k], dim=0)
        if to_w:
            latents_view_dict[k] = latent_space_ops.wplus_to_w(latents_view_dict[k])

    return latents, latents_exp_dict, latents_view_dict, latents_exp_dict_dict, latents_view_dict_dict



def load_alpha_deltas(alpha_dir: Path, deltas_dir: Path):
    '''
    alpha and deltas are generated during test image projection.
    alpha: [N, 1], where N is the number of anchors
    deltas: 
    '''
    alpha_deltas_dict = {}
    
    for f in alpha_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        image_name = str(f).split('/alpha_opt/')[-1].split('.')[0]
        alpha = torch.load(f)
        alpha_deltas_dict[image_name] = {'alpha': alpha}
        
    for f in deltas_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        image_name = str(f).split('/deltas_opt/')[-1].split('.')[0]
        deltas = torch.load(f)
        if image_name in alpha_deltas_dict.keys():
            alpha_deltas_dict[image_name]['deltas'] = deltas
        else:
            print('No alpha found for {}'.format(image_name))
        
    return alpha_deltas_dict


def load_latents_coord_access_vyvp(latents_dir: Path, to_w=False):
    '''
    anchor loss of PCA applied.
    load latents for coordinate access.
    view is represented using yaw and pitch.
    '''
    latents = []
    latents_exp_dict = {}
    latents_view_dict = {}
    latents_exp_dict_dict = {}
    latents_view_dict_dict = {}
    
    # import pdb; pdb.set_trace()

    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp_name, viewy_name, viewp_name = image_name_tokens[-2], image_name_tokens[1], image_name_tokens[2]
        view_name = viewy_name + '_' + viewp_name
        latent_pt = torch.load(f)
        
        latents.append(latent_pt)

        if exp_name in latents_exp_dict.keys():
            latents_exp_dict[exp_name].append(latent_pt)
            latents_exp_dict_dict[exp_name][view_name] = latent_pt
        else:
            latents_exp_dict[exp_name] = [latent_pt]
            latents_exp_dict_dict[exp_name] = {view_name: latent_pt}

        if view_name in latents_view_dict.keys():
            latents_view_dict[view_name].append(latent_pt)
            latents_view_dict_dict[view_name][exp_name] = latent_pt
        else:
            latents_view_dict[view_name] = [latent_pt]
            latents_view_dict_dict[view_name] = {exp_name: latent_pt}

    latents = torch.stack(latents, dim=0)
    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    for k in latents_exp_dict.keys():
        latents_exp_dict[k]= torch.stack(latents_exp_dict[k], dim=0)
        if to_w:
            latents_exp_dict[k] = latent_space_ops.wplus_to_w(latents_exp_dict[k])

    for k in latents_view_dict.keys():
        latents_view_dict[k]= torch.stack(latents_view_dict[k], dim=0)
        if to_w:
            latents_view_dict[k] = latent_space_ops.wplus_to_w(latents_view_dict[k])

    return latents, latents_exp_dict, latents_view_dict, latents_exp_dict_dict, latents_view_dict_dict



def load_latents_fb_multifaces_view_pair(latents_dir: Path, view_list, exp_list, to_w=False):
    latents_dict = {}
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue
        
        # image_name_tokens, ['figure0', 'E001', 'Neutral', 'Eyes', 'Open', '400004', '009357.pt']
        image_name_tokens = str(f).split('/w/')[-1].split('_')
        exp, frame = image_name_tokens[1], image_name_tokens[-2]
        latent_pt = torch.load(f)        
        if frame in latents_dict.keys():
            latents_dict[frame][exp] = latent_pt
        else:
            latents_dict[frame] = {exp: latent_pt}
            
    latents_view_dict = {}
    for view in view_list:
        view_latents = []
        for exp in exp_list:
            view_latents.append(latents_dict[view][exp])
            
        view_latents = torch.stack(view_latents, dim=0)
        if to_w:
            view_latents = latent_space_ops.wplus_to_w(view_latents)
        latents_view_dict[view] = view_latents
    
    return latents_view_dict


def load_net(file_path: Path):
    try:
        with dnnlib.util.open_url(str(file_path)) as f:
            G = pickle.load(f)['G_ema'].synthesis
    except Exception as e:
        G = torch.load(file_path)

    return G.cuda()


def save_images(frames: torch.FloatTensor, output_path: Path):
    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)

    torchvision.utils.save_image(
        frames,
        output_path.with_suffix('.jpg'),
        nrow=frames.shape[0],
        normalize=True,
        range=(-1, 1)
    )


def save_latents(latent: torch.FloatTensor, output_path: Path):
    if latent is None:
        return

    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)
    torch.save(latent, output_path)


def load_mask(mask_path: Path):
    mask_img = Image.open(mask_path).convert('L')
    mask = np.array(mask_img)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0

    mask = torch.FloatTensor(mask)
    mask = torch.unsqueeze(mask, dim=0) / 255
    return mask


def get_images_in_dir(input_dir: Path):
    global IMAGE_SUFFIX
    image_fps = [fp for fp in input_dir.iterdir() if fp.suffix in IMAGE_SUFFIX]
    return image_fps
