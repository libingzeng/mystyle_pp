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

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from utils import latent_space_ops, io_utils

from torchvision.utils import save_image
import torch
import numpy as np
from utils.data_utils import PersonalizedDataset
import imageio

torch.manual_seed(2)
np.random.seed(2)

# required for pickle to magically find torch_utils for loading official FFHQ checkpoint
sys.path.append('third_party/stylegan2_ada_pytorch')

def synthesize_test_pca_projected_anchor_auto_yawpitchexpage(anchors_path, anchors_test_path, generator_path, output_path, axis_path, align, has_age, alg_type):
    generator = io_utils.load_net(generator_path).to('cuda')
    from sklearn.decomposition import PCA
    
    if os.path.exists(axis_path):
        axis_dict = torch.load(axis_path)
        yaw_axis = axis_dict['yaw_axis']
        exp_axis = axis_dict['exp_axis']
        pitch_axis = axis_dict['pitch_axis']
        if has_age != 0:
            age_axis = axis_dict['age_axis']
    else:
        yaw_axis = 0
        exp_axis = 1
        pitch_axis = 2
        if has_age != 0:
            age_axis = 3

    pti_image_name = str(generator_path).split('/')[-2]

    anchors_all = \
        io_utils.load_latents(anchors_path)[:, :, :1, :]
    anchor_test_pti_path = anchors_test_path.joinpath('{}.pt'.format(pti_image_name))
    anchor_test_pti = io_utils.load_single_latent(anchor_test_pti_path)[:, 0:1, :]
    
    import time
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    
    pca = PCA(n_components=0.999999999)
    anchors_proj = pca.fit_transform(X.cpu().numpy())
    latent = X.reshape(anchors_all.shape).squeeze()
    latent_inv  = torch.from_numpy(pca.inverse_transform(anchors_proj).reshape(anchors_all.shape).squeeze()).cuda()
    
    pc_y_min, pc_y_max = anchors_proj[:, yaw_axis].min(), anchors_proj[:, yaw_axis].max()
    pc_e_min, pc_e_max = anchors_proj[:, exp_axis].min(), anchors_proj[:, exp_axis].max()
    pc_p_min, pc_p_max = anchors_proj[:, pitch_axis].min(), anchors_proj[:, pitch_axis].max()
    if has_age != 0:
        pc_a_min, pc_a_max = anchors_proj[:, age_axis].min(), anchors_proj[:, age_axis].max()

    output_path.joinpath('coord_select_strategy4_pc_10samples_final').mkdir(exist_ok=True, parents=True)
    output_images_path = output_path.joinpath('coord_select_strategy4_pc_10samples_final', 'images')
    output_images_path.mkdir(exist_ok=True, parents=True)
    output_video_path = output_path.joinpath('coord_select_strategy4_pc_10samples_final', 'videos')
    output_video_path.mkdir(exist_ok=True, parents=True)

    yaw_num, exp_num, pitch_num, age_num = 21, 21, 21, 21
    minend_y, maxend_y = 0.0, yaw_num
    minend_p, maxend_p = 0.0, exp_num
    minend_e, maxend_e = 0.0, pitch_num
    if has_age != 0:
        minend_a, maxend_a = 0.0, age_num

    celebrity_name = str(output_path).split('/output_')[0].split('/')[-1]
    align_dict = {\
        'barack_obama':{
            'ours': {'yaw':{'start': 7, 'end':15}, \
                            'pitch':{'start': 7, 'end':12}, \
                            'exp':{'start': 4, 'end':14}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': ['2422304508_7a6490730a_o'], \
                    }, \
            'original': {'yaw':{'start': 3, 'end':16}, \
                            'pitch':{'start': 4, 'end':15}, \
                            'exp':{'start': 5, 'end':11}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': ['2422304508_7a6490730a_o'], \
                    }, \
            }, \
        'joe_biden':{
            'ours': {'yaw':{'start': 5, 'end':11}, \
                            'pitch':{'start': 3, 'end':11}, \
                            'exp':{'start': 4, 'end':13}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            'original': {'yaw':{'start': 4, 'end':18}, \
                            'pitch':{'start': 0, 'end':14}, \
                            'exp':{'start': 0, 'end':20}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            }, \
        'michelle_obama':{
            'ours': {'yaw':{'start': 5, 'end':17}, \
                            'pitch':{'start': 4, 'end':9}, \
                            'exp':{'start': 3, 'end':14}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': ['30343162332_da82964d93_o'], \
                    }, \
            'original': {'yaw':{'start': 4, 'end':17}, \
                            'pitch':{'start': 4, 'end':17}, \
                            'exp':{'start': 3, 'end':14}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': ['30343162332_da82964d93_o'], \
                    }, \
            }, \
        'oprah_winfrey':{
            'ours': {'yaw':{'start': 0, 'end':16}, \
                            'pitch':{'start': 0, 'end':10}, \
                            'exp':{'start': 0, 'end':10}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            'original': {'yaw':{'start': 0, 'end':16}, \
                            'pitch':{'start': 0, 'end':10}, \
                            'exp':{'start': 0, 'end':10}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            }, \
        'scarlett_johansson':{
            'ours': {'yaw':{'start': 3, 'end':13}, \
                            'pitch':{'start': 4, 'end':13}, \
                            'exp':{'start': 4, 'end':15}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            'original': {'yaw':{'start': 1, 'end':19}, \
                            'pitch':{'start': 0, 'end':20}, \
                            'exp':{'start': 7, 'end':19}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            }, \
        'scarlett_johansson46':{
            'ours': {'yaw':{'start': 0, 'end':17}, \
                            'pitch':{'start': 2, 'end':18}, \
                            'exp':{'start': 0, 'end':13}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            'original': {'yaw':{'start': 1, 'end':19}, \
                            'pitch':{'start': 3, 'end':17}, \
                            'exp':{'start': 7, 'end':15}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            }, \
        'scarlett_johansson92':{
            'ours': {'yaw':{'start': 0, 'end':7}, \
                            'pitch':{'start': 3, 'end':17}, \
                            'exp':{'start': 8, 'end':15}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            'original': {'yaw':{'start': 1, 'end':19}, \
                            'pitch':{'start': 0, 'end':20}, \
                            'exp':{'start': 7, 'end':19}, \
                            'age':{'start': 0, 'end':0}, \
                            'magnitude_scale': 2, \
                            'test_list': [], \
                    }, \
            }, \
        'leonardo_dicaprio':{
            'ours': {'yaw':{'start': 2, 'end':13}, \
                            'pitch':{'start': 3, 'end':6}, \
                            'exp':{'start': 4, 'end':10}, \
                            'age':{'start': 0, 'end':6}, \
                            'magnitude_scale': 2, \
                            'test_list': ['11933209533_b0df7d2240_o'], \
                    }, \
            'original': {'yaw':{'start': 4, 'end':16}, \
                            'pitch':{'start': 2, 'end':20}, \
                            'exp':{'start': 3, 'end':17}, \
                            'age':{'start': 8, 'end':15}, \
                            'magnitude_scale': 2, \
                            'test_list': ['11933209533_b0df7d2240_o'], \
                    }, \
            }, \
        'emma_watson':{
            'ours': {'yaw':{'start': 3, 'end':12}, \
                            'pitch':{'start': 3, 'end':13}, \
                            'exp':{'start': 5, 'end':11}, \
                            'age':{'start': 4, 'end':11}, \
                            'magnitude_scale': 2, \
                            'test_list': ['Emma_Watson_Cannes_2013_3'], \
                    }, \
            'original': {'yaw':{'start': 1, 'end':15}, \
                            'pitch':{'start': 4, 'end':16}, \
                            'exp':{'start': 1, 'end':14}, \
                            'age':{'start': 2, 'end':19}, \
                            'magnitude_scale': 4, \
                            'test_list': ['Emma_Watson_Cannes_2013_3'], \
                    }, \
            }
        }
    
    magnitude_scale = align_dict[celebrity_name][alg_type]['magnitude_scale']
    if len(align_dict[celebrity_name][alg_type]['test_list']) > 0:
        if pti_image_name not in align_dict[celebrity_name][alg_type]['test_list']:
            return

    if align != 0:
        # import pdb; pdb.set_trace()
        minend_y, maxend_y = align_dict[celebrity_name][alg_type]['yaw']['start'], align_dict[celebrity_name][alg_type]['yaw']['end']
        minend_p, maxend_p = align_dict[celebrity_name][alg_type]['pitch']['start'], align_dict[celebrity_name][alg_type]['pitch']['end']
        minend_e, maxend_e = align_dict[celebrity_name][alg_type]['exp']['start'], align_dict[celebrity_name][alg_type]['exp']['end']
        if has_age != 0:
            minend_a, maxend_a = align_dict[celebrity_name][alg_type]['age']['start'], align_dict[celebrity_name][alg_type]['age']['end']

        
        output_images_path = output_path.joinpath('coord_select_strategy4_pc_10samples_final', 'images_aligned_{}'.format(magnitude_scale))
        output_images_path.mkdir(exist_ok=True, parents=True)
        output_video_path = output_path.joinpath('coord_select_strategy4_pc_10samples_final', 'videos_aligned_{}'.format(magnitude_scale))
        output_video_path.mkdir(exist_ok=True, parents=True)
    
    anchor = anchor_test_pti.reshape(1, -1).cpu().numpy()
    anchor_proj = pca.transform(anchor)
    anchor_proj_yaw_cp = np.copy(anchor_proj)
    anchor_proj_exp_cp = np.copy(anchor_proj)
    anchor_proj_pitch_cp = np.copy(anchor_proj)
    yaw_num, exp_num, pitch_num, age_num = 21, 21, 21, 21
    
    video_yaw = imageio.get_writer(output_path.joinpath(output_video_path, '{}_yaw_editing.mp4'.format(pti_image_name)), mode='I', fps=6, codec='libx264', bitrate='16M')
    video_exp = imageio.get_writer(output_path.joinpath(output_video_path, '{}_exp_editing.mp4'.format(pti_image_name)), mode='I', fps=6, codec='libx264', bitrate='16M')
    video_pitch = imageio.get_writer(output_path.joinpath(output_video_path, '{}_pitch_editing.mp4'.format(pti_image_name)), mode='I', fps=6, codec='libx264', bitrate='16M')
    if has_age != 0:
        anchor_proj_age_cp = np.copy(anchor_proj)
        video_age = imageio.get_writer(output_path.joinpath(output_video_path, '{}_age_editing.mp4'.format(pti_image_name)), mode='I', fps=6, codec='libx264', bitrate='16M')

    for y in range(yaw_num):
        y_aligned = (maxend_y - minend_y) / yaw_num * y + minend_y
        yaw_coord = np.clip(1 / yaw_num * y_aligned, 0.0, 1.0)
        y_pc = (yaw_coord * (pc_y_max - pc_y_min) + pc_y_min) * magnitude_scale
        
        print('img:{}, yaw_editing: {}'.format(pti_image_name, y))
        anchor_proj_yaw_cp[0][yaw_axis] = y_pc
        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_yaw_cp).reshape(anchor.shape).squeeze()).cuda()
        img_inv = generator(latent_inv.expand(1, 18, 512), noise_mode='const', force_fp32=True)[0]
        video_yaw.append_data(((img_inv + 1) / 2 * 255).clamp(0,255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        save_image(img_inv, output_images_path.joinpath('{}_yaw_editing_{}.jpg'.format(pti_image_name, y)), nrow=1, normalize=True, range=(-1, 1))
        del img_inv
    video_yaw.close()
    
    for p in range(pitch_num):
        p_aligned = (maxend_p - minend_p) / pitch_num * p + minend_p
        pitch_coord = np.clip(1 / pitch_num * p_aligned, 0.0, 1.0)
        p_pc = (pitch_coord * (pc_p_max - pc_p_min) + pc_p_min) * magnitude_scale

        print('img:{}, pitch_editing: {}'.format(pti_image_name, p))
        anchor_proj_pitch_cp[0][pitch_axis] = p_pc
        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_pitch_cp).reshape(anchor.shape).squeeze()).cuda()
        img_inv = generator(latent_inv.expand(1, 18, 512), noise_mode='const', force_fp32=True)[0]
        video_pitch.append_data(((img_inv + 1) / 2 * 255).clamp(0,255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        save_image(img_inv, output_images_path.joinpath('{}_pitch_editing_{}.jpg'.format(pti_image_name, p)), nrow=1, normalize=True, range=(-1, 1))
        del img_inv
    video_pitch.close()

    for e in range(exp_num):
        e_aligned = (maxend_e - minend_e) / exp_num * e + minend_e
        exp_coord = np.clip(1 / exp_num * e_aligned, 0.0, 1.0)
        e_pc = (exp_coord * (pc_e_max - pc_e_min) + pc_e_min) * magnitude_scale
            
        print('img:{}, exp_editing: {}'.format(pti_image_name, e))
        anchor_proj_exp_cp[0][exp_axis] = e_pc
        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_exp_cp).reshape(anchor.shape).squeeze()).cuda()
        img_inv = generator(latent_inv.expand(1, 18, 512), noise_mode='const', force_fp32=True)[0]
        video_exp.append_data(((img_inv + 1) / 2 * 255).clamp(0,255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        save_image(img_inv, output_images_path.joinpath('{}_exp_editing_{}.jpg'.format(pti_image_name, e)), nrow=1, normalize=True, range=(-1, 1))
        del img_inv
    video_exp.close()
    
    if has_age != 0:
        for a in range(age_num):
            a_aligned = (maxend_a - minend_a) / age_num * a + minend_a
            age_coord = np.clip(1 / age_num * a_aligned, 0.0, 1.0)
            a_pc = (age_coord * (pc_a_max - pc_a_min) + pc_a_min) * magnitude_scale

            print('img:{}, age_editing: {}'.format(pti_image_name, a))
            anchor_proj_age_cp[0][age_axis] = a_pc
            latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_age_cp).reshape(anchor.shape).squeeze()).cuda()
            img_inv = generator(latent_inv.expand(1, 18, 512), noise_mode='const', force_fp32=True)[0]
            video_age.append_data(((img_inv + 1) / 2 * 255).clamp(0,255).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            save_image(img_inv, output_images_path.joinpath('{}_age_editing_{}.jpg'.format(pti_image_name, a)), nrow=1, normalize=True, range=(-1, 1))
            del img_inv
        video_age.close()


def parse_args(raw_args):
    parser = ArgumentParser()
    parser.add_argument('--images_dir', required=True, type=Path)
    parser.add_argument('--anchor_dir', required=True, type=Path)
    parser.add_argument('--pti_images_dir', required=True, type=Path)
    parser.add_argument('--output_path', required=True, type=Path)
    parser.add_argument('--anchors_path', required=True, type=Path)
    parser.add_argument('--axis_path', required=False, type=Path)

    parser.add_argument('--device', default='0')
    parser.add_argument('--align', type=int, default=0)
    parser.add_argument('--has_age', type=int, default=0)
    parser.add_argument('--alg_type', type=str, default='ours', help='ours, or original')

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.output_path.mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('latents').mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('images').mkdir(exist_ok=True, parents=True)

    return args


def get_data(args):
    if args.anchor_dir is not None:
        dataset = PersonalizedDataset(args.images_dir, args.anchor_dir)
    else:
        print('No fine-tuned latents found.')
    return dataset


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)
    
    dataset = get_data(args)

    for sample in dataset:
        print('Image editing for image: {}'.format(sample.name))
        
        args.pti_sample_dir = args.pti_images_dir.joinpath('{}'.format(sample.name))
        args.generator_path = args.pti_sample_dir.joinpath(f'mystyle_model.pt')
        
        synthesize_test_pca_projected_anchor_auto_yawpitchexpage(args.anchors_path, args.anchor_dir, args.generator_path, args.output_path, args.axis_path, args.align, args.has_age, args.alg_type)

if __name__ == '__main__':
    with torch.no_grad():
        main()
