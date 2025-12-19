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

torch.manual_seed(2)
np.random.seed(2)

# required for pickle to magically find torch_utils for loading official FFHQ checkpoint
sys.path.append('third_party/stylegan2_ada_pytorch')


def synthesize(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    anchors = io_utils.load_latents(anchors_path)
    # anchors = io_utils.load_latents(anchors_path)[:10] # first 10 axes (single expression test)
    latents = latent_space_ops.sample_from_P0(anchors, num_points_to_sample, num_anchor_points).to('cuda')
    generator = io_utils.load_net(generator_path).to('cuda')

    if latents.shape[2] != 18:
        latents = latents.repeat(1, 1, 18, 1)

    batch_size = 4
    i = 0
    while i < latents.shape[0]:
        lats = latents[i: i + batch_size]
        imgs = generator(lats.squeeze(1), noise_mode='const', force_fp32=True)

        for j in range(min(batch_size, num_points_to_sample - i)):
            save_image(imgs[j], output_path.joinpath('images', f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))
            io_utils.save_latents(lats[j], output_path.joinpath('latents', f'{i + j}.pt'))

        del imgs
        i += batch_size


def synthesize_fb_multifaces(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    anchors_all, anchors_exp_dict, anchors_frame_dict = io_utils.load_latents_fb_multifaces(anchors_path)
    
    gen_modes = {'0':'single_expression', \
                '1': 'single_frame', \
                }
    mode_key = '0'
    if gen_modes[mode_key] == 'single_expression':
        for exp in anchors_exp_dict.keys():
            # pdb.set_trace()
            output_path.joinpath(exp).mkdir(exist_ok=True, parents=True)
            output_path.joinpath(exp, 'images').mkdir(exist_ok=True, parents=True)
            output_path.joinpath(exp, 'latents').mkdir(exist_ok=True, parents=True)

            latents = latent_space_ops.sample_from_P0(anchors_exp_dict[exp], num_points_to_sample, num_anchor_points).to('cuda')
            
            batch_size = 4
            i = 0
            while i < latents.shape[0]:
                lats = latents[i: i + batch_size]
                pdb.set_trace()
                imgs = generator(lats.squeeze(1), noise_mode='const', force_fp32=True)

                for j in range(min(batch_size, num_points_to_sample - i)):
                    save_image(imgs[j], output_path.joinpath(exp, 'images', f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))
                    io_utils.save_latents(lats[j], output_path.joinpath(exp, 'latents', f'{i + j}.pt'))

                del imgs
                i += batch_size

    if gen_modes[mode_key] == 'single_frame':
        for frame in anchors_frame_dict.keys():
            # pdb.set_trace()
            output_path.joinpath(frame).mkdir(exist_ok=True, parents=True)
            output_path.joinpath(frame, 'images').mkdir(exist_ok=True, parents=True)
            output_path.joinpath(frame, 'latents').mkdir(exist_ok=True, parents=True)

            latents = latent_space_ops.sample_from_P0(anchors_frame_dict[frame], num_points_to_sample, num_anchor_points).to('cuda')
            
            batch_size = 4
            i = 0
            while i < latents.shape[0]:
                lats = latents[i: i + batch_size]
                imgs = generator(lats.squeeze(1), noise_mode='const', force_fp32=True)

                for j in range(min(batch_size, num_points_to_sample - i)):
                    save_image(imgs[j], output_path.joinpath(frame, 'images', f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))
                    io_utils.save_latents(lats[j], output_path.joinpath(frame, 'latents', f'{i + j}.pt'))

                del imgs
                i += batch_size


def synthesize_fb_multifaces_pca_plots(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    anchors_all, anchors_exp_dict, anchors_frame_dict = io_utils.load_latents_fb_multifaces(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    U, S, V = torch.pca_lowrank(X)
    # low-dimensional reconstruction
    rd = 2 # number of reduced dimensions to use
    X_red = U[:, :rd] @ (torch.diag(S)[:rd, :rd]) @ V.T[:rd, :]
    X_proj = torch.matmul(X, V[:, :rd])
    anchors_proj = X_proj.cpu().numpy()
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()
    import matplotlib.pyplot as plt
    color_list = ['yellow', 'black', 'green', 'orange', 'navy', 'pink', 'purple', 'red', \
                'sienna', 'salmon', 'dimgray', 'silver', 'springgreen', 'aquamarine', 'teal', 'cyan', \
                'violet', 'blue', 'cornflowerblue', 'plum', 'purple', 'palevioletred', 'greenyellow', 'wheat', \
                ]
    cnt = 0
    output_path.joinpath('plots').mkdir(exist_ok=True, parents=True)
    # for i in range(len(anchors_proj)): 
    #     plt.scatter(anchors_proj[i][0], anchors_proj[i][1])
    
    is_averaged = False
    is_averaged_only = False
    plot_attr_dict = {0:'exp', 1:'view'}
    plot_attr_key = 1
    
    if is_averaged_only:
        scatter_x, scatter_y = [], []
        for exp_key in anchors_exp_dict.keys():
            anchors_exp = anchors_exp_dict[exp_key]
            anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
            anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd]).cpu().numpy()
            scatter_x.append((anchors_exp_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
            scatter_y.append((anchors_exp_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
        plt.scatter(scatter_x, scatter_y, color='black', label='exp')
        scatter_x, scatter_y = [], []
        for view_key in anchors_frame_dict.keys():
            anchors_view = anchors_frame_dict[view_key]
            anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
            anchors_view_proj = torch.matmul(anchors_view, V[:, :rd]).cpu().numpy()
            scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
            scatter_y.append((anchors_view_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
        plt.scatter(scatter_x, scatter_y, color='pink', label='view')
        plt.xlabel('PCA C1 View')
        plt.ylabel('PCA C2 Exp')
        plt.legend(loc='upper right', fontsize=6)
        plt.title('Average Exp, Average View')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.savefig(output_path.joinpath('plots', 'anchors_pca2_exp_view_avg.png'))

    else: # is_averaged_only:
        if plot_attr_dict[plot_attr_key] == 'exp':
            for exp_key in anchors_exp_dict.keys():
                anchors_exp = anchors_exp_dict[exp_key]
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd]).cpu().numpy()
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_exp_proj)): 
                        scatter_x.append((anchors_exp_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_exp_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_exp_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_exp_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=exp_key)
                cnt += 1
            plt.xlabel('PCA C1')
            plt.ylabel('PCA C2')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('View Distribution of Anchors PCA2')
                plt.savefig(output_path.joinpath('plots', 'anchors_pca2_exp.png'))
            else:
                plt.title('View Average of Anchors PCA2')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots', 'anchors_pca2_exp_avg.png'))

        if plot_attr_dict[plot_attr_key] == 'view':
            for view_key in anchors_frame_dict.keys():
                anchors_view = anchors_frame_dict[view_key]
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd]).cpu().numpy()
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_view_proj)): 
                        scatter_x.append((anchors_view_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_view_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_view_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=view_key)
                cnt += 1
            plt.xlabel('PCA C1')
            plt.ylabel('PCA C2')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('Expression Distribution of Anchors PCA1')
                plt.savefig(output_path.joinpath('plots', 'anchors_pca1_view.png'))
            else:
                plt.title('Expression Average of Anchors PCA1')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots', 'anchors_pca1_view_avg.png'))
    pdb.set_trace()


def synthesize_fb_multifaces_pca_plots_sklearn(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    # import pdb; pdb.set_trace()
    anchors_all, anchors_exp_dict, anchors_frame_dict = io_utils.load_latents_fb_multifaces(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99)
    anchors_proj = pca.fit_transform(X.cpu().numpy())

    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()
    import matplotlib.pyplot as plt
    color_list = ['yellow', 'black', 'green', 'orange', 'navy', 'pink', 'purple', 'red', \
                'sienna', 'salmon', 'dimgray', 'silver', 'springgreen', 'aquamarine', 'teal', 'cyan', \
                'violet', 'blue', 'cornflowerblue', 'plum', 'purple', 'palevioletred', 'greenyellow', 'wheat', \
                'tan', 'lightblue', 'slateblue', 'olive', 'linen', 'mediumvioletred', 'crimson', 'fuchsia', \
                'chocolate', 'tomato', 'rosybrown', 'khaki', 'indianred', 'darkseegreen'\
                ]
    cnt = 0
    output_path.joinpath('plots_sklearn').mkdir(exist_ok=True, parents=True)
    # for i in range(len(anchors_proj)): 
    #     plt.scatter(anchors_proj[i][0], anchors_proj[i][1])
    
    is_averaged = False
    is_averaged_only = False
    plot_attr_dict = {0:'exp', 1:'view'}
    plot_attr_key = 1
    
    if is_averaged_only:
        scatter_x, scatter_y = [], []
        for exp_key in anchors_exp_dict.keys():
            anchors_exp = anchors_exp_dict[exp_key]
            anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
            anchors_exp_proj = pca.transform(anchors_exp)
            scatter_x.append((anchors_exp_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
            scatter_y.append((anchors_exp_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
        plt.scatter(scatter_x, scatter_y, color='black', label='exp')
        scatter_x, scatter_y = [], []
        for view_key in anchors_frame_dict.keys():
            anchors_view = anchors_frame_dict[view_key]
            anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
            anchors_exp_proj = pca.transform(anchors_view)
            scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
            scatter_y.append((anchors_view_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
        plt.scatter(scatter_x, scatter_y, color='pink', label='view')
        plt.xlabel('View Component')
        plt.ylabel('Expression Component')
        plt.legend(loc='upper right', fontsize=6)
        plt.title('Average Exp, Average View')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_pca2_exp_view_avg.png'))

    else: # is_averaged_only:
        if plot_attr_dict[plot_attr_key] == 'exp':
            for exp_key in anchors_exp_dict.keys():
                anchors_exp = anchors_exp_dict[exp_key]
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1).cpu().numpy()
                anchors_exp_proj = pca.transform(anchors_exp)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_exp_proj)): 
                        scatter_x.append((anchors_exp_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_exp_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_exp_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_exp_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=exp_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Expression Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by expressions)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_pca2_exp.png'))
            else:
                plt.title('View Average of Anchors PCA2')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_pca2_exp_avg.png'))

        if plot_attr_dict[plot_attr_key] == 'view':
            for view_key in anchors_frame_dict.keys():
                anchors_view = anchors_frame_dict[view_key]
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1).cpu().numpy()
                anchors_view_proj = pca.transform(anchors_view)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_view_proj)): 
                        scatter_x.append((anchors_view_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_view_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_view_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=view_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Expression Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by views)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_pca1_view.png'))
            else:
                plt.title('Expression Average of Anchors PCA1')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_pca1_view_avg.png'))


def synthesize_fb_multifaces_pca_plots_sklearn_lightstage(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    # import pdb; pdb.set_trace()
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_light_dict = io_utils.load_latents_lightstage(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99)
    anchors_proj = pca.fit_transform(X.cpu().numpy())

    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max() # view
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max() # expression
    pc2_min, pc2_max = anchors_proj[:, 2].min(), anchors_proj[:, 2].max() # lighting
    import matplotlib.pyplot as plt
    color_list = ['yellow', 'black', 'green', 'orange', 'navy', 'pink', 'purple', 'red', \
                'sienna', 'salmon', 'dimgray', 'silver', 'springgreen', 'aquamarine', 'teal', 'cyan', \
                'violet', 'blue', 'cornflowerblue', 'plum', 'purple', 'palevioletred', 'greenyellow', 'wheat', \
                'tan', 'lightblue', 'slateblue', 'olive', 'linen', 'mediumvioletred', 'crimson', 'fuchsia', \
                'chocolate', 'tomato', 'rosybrown', 'khaki', 'indianred', 'darkseegreen'\
                ]
    cnt = 0
    output_path.joinpath('plots_sklearn').mkdir(exist_ok=True, parents=True)
    # for i in range(len(anchors_proj)): 
    #     plt.scatter(anchors_proj[i][0], anchors_proj[i][1])
    
    is_averaged = False
    plot_attr_dict = {0:'exp', 1:'view', 2:'light'}
    plot_mode_dict = {0:'view-exp', 1:'view-light'}
    plot_attr_key, plot_mode_key = 2, 1 # options: (0, 0), (1, 0); (2, 1), (1, 1)
    
    if plot_mode_dict[plot_mode_key] == 'view-exp':
        if plot_attr_dict[plot_attr_key] == 'exp':
            for exp_key in anchors_exp_dict.keys():
                anchors_exp = anchors_exp_dict[exp_key]
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1).cpu().numpy()
                anchors_exp_proj = pca.transform(anchors_exp)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_exp_proj)): 
                        scatter_x.append((anchors_exp_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_exp_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_exp_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_exp_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=exp_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Expression Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by expressions)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-exp_exp.png'))
            else:
                plt.title('View Average of Anchors PCA2')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-exp_exp_avg.png'))

        if plot_attr_dict[plot_attr_key] == 'view':
            for view_key in anchors_frame_dict.keys():
                anchors_view = anchors_frame_dict[view_key]
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1).cpu().numpy()
                anchors_view_proj = pca.transform(anchors_view)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_view_proj)): 
                        scatter_x.append((anchors_view_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_view_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
                else:
                    scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_view_proj.mean(0)[1] - pc1_min) / (pc1_max - pc1_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=view_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Expression Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by views)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-exp_view.png'))
            else:
                plt.title('Expression Average of Anchors PCA1')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-exp_view_avg.png'))
    if plot_mode_dict[plot_mode_key] == 'view-light':
        if plot_attr_dict[plot_attr_key] == 'light':
            for light_key in anchors_light_dict.keys():
                anchors_light = anchors_light_dict[light_key]
                anchors_light = anchors_light.reshape(anchors_light.shape[0], -1).cpu().numpy()
                anchors_light_proj = pca.transform(anchors_light)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_light_proj)): 
                        scatter_x.append((anchors_light_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_light_proj[i][2] - pc2_min) / (pc2_max - pc2_min))
                else:
                    scatter_x.append((anchors_light_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_light_proj.mean(0)[2] - pc2_min) / (pc2_max - pc2_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=light_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Light Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by lightressions)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-light_light.png'))
            else:
                plt.title('View Average of Anchors PCA2')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-light_light_avg.png'))

        if plot_attr_dict[plot_attr_key] == 'view':
            for view_key in anchors_frame_dict.keys():
                anchors_view = anchors_frame_dict[view_key]
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1).cpu().numpy()
                anchors_view_proj = pca.transform(anchors_view)
                scatter_x, scatter_y = [], []
                if not is_averaged:
                    for i in range(len(anchors_view_proj)): 
                        scatter_x.append((anchors_view_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
                        scatter_y.append((anchors_view_proj[i][2] - pc2_min) / (pc2_max - pc2_min))
                else:
                    scatter_x.append((anchors_view_proj.mean(0)[0] - pc0_min) / (pc0_max - pc0_min))
                    scatter_y.append((anchors_view_proj.mean(0)[2] - pc2_min) / (pc2_max - pc2_min))
                plt.scatter(scatter_x, scatter_y, color=color_list[cnt], label=view_key)
                cnt += 1
            plt.xlabel('View Component')
            plt.ylabel('Light Component')
            plt.legend(loc='upper right', fontsize=6)
            if not is_averaged:
                plt.title('PCA of Latent Codes (colored by views)')
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-light_view.png'))
            else:
                plt.title('Light Average of Anchors PCA1')
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.savefig(output_path.joinpath('plots_sklearn', 'anchors_view-light_view_avg.png'))

    

def synthesize_fb_multifaces_pca(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    import pandas as pd
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    U, S, V = torch.pca_lowrank(X)
    pdb.set_trace()
    # low-dimensional reconstruction
    rd = 2 # number of reduced dimensions to use
    X_proj = torch.matmul(X, V[:, :rd])
    anchors_proj = X_proj.cpu().numpy()
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    view_names, view_anchors_avgs, view_anchors_avg_norms = [], [], []
    for view_key in anchors_frame_dict.keys():
        anchors_view = anchors_frame_dict[view_key]
        anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
        anchors_view_proj = torch.matmul(anchors_view, V[:, :rd]).cpu().numpy()
        anchor_pc0 = anchors_view_proj[:, 0].mean()
        anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
        view_names.append(view_key)
        view_anchors_avgs.append(anchor_pc0)
        view_anchors_avg_norms.append(anchor_pc0_norm)
    data_view = {
        'view_anchors_avg': view_anchors_avgs, \
        'view_anchors_avg_norm': view_anchors_avg_norms, \
        }
    df_view = pd.DataFrame(data_view, index=view_names)

    exp_names, exp_anchors_avgs, exp_anchors_avg_norms = [], [], []
    for exp_key in anchors_exp_dict.keys():
        anchors_exp = anchors_exp_dict[exp_key]
        anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
        anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd]).cpu().numpy()
        anchor_pc1 = anchors_exp_proj[:, 1].mean()
        anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
        exp_names.append(exp_key)
        exp_anchors_avgs.append(anchor_pc1)
        exp_anchors_avg_norms.append(anchor_pc1_norm)
    data_exp = {
        'exp_anchors_avg': exp_anchors_avgs, \
        'exp_anchors_avg_norm': exp_anchors_avg_norms, \
        }
    df_exp = pd.DataFrame(data_exp, index=exp_names)

    anchor_names, anchor_pc0s, anchor_pc1s, anchor_pc0_norms, anchor_pc1_norms = [], [], [], [], []
    anchor_view_names, anchor_exp_names = [], []
    for view_key in anchors_frame_dict_dict.keys():
        for exp_key in anchors_frame_dict_dict[view_key].keys():
            anchor_name = view_key + '_' + exp_key
            anchor = anchors_frame_dict_dict[view_key][exp_key]
            pdb.set_trace()
            anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd]).cpu().numpy()
            anchor_pc0 = anchor_proj[0][0]
            anchor_pc1 = anchor_proj[0][1]
            anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
            anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
            anchor_names.append(anchor_name)
            anchor_pc0s.append(anchor_pc0)
            anchor_pc1s.append(anchor_pc1)
            anchor_pc0_norms.append(anchor_pc0_norm)
            anchor_pc1_norms.append(anchor_pc1_norm)
            anchor_view_names.append(view_key)
            anchor_exp_names.append(exp_key)
    data = {
        'anchor_view_name': anchor_view_names, \
        'anchor_exp_name': anchor_exp_names, \
        'anchor_pc0': anchor_pc0s, \
        'anchor_pc1': anchor_pc1s, \
        'anchor_pc0_norm': anchor_pc0_norms, \
        'anchor_pc1_norm': anchor_pc1_norms
        }
    df = pd.DataFrame(data, index=anchor_names)
    
    view_num, exp_num = 50, 50
    for v in range(view_num):
        for e in range(exp_num):
            # view_coord, exp_coord = np.clip(1 / view_num * v, 0.01, 0.99), np.clip(1 / exp_num * e, 0.01, 0.99)
            view_coord, exp_coord = 0.3, 0.6

            # select two views bounding the given view_coord
            df_view_larger = df_view.loc[(df_view['view_anchors_avg_norm'] - view_coord) > 0]
            if not df_view_larger.empty:
                view_select_larger = df_view_larger.iloc[df_view_larger['view_anchors_avg_norm'].argmin()].name
            else:
                view_select_larger = df_view.iloc[df_view['view_anchors_avg_norm'].argmax()].name
            df_view_less = df_view.loc[(df_view['view_anchors_avg_norm'] - view_coord) <= 0]
            if not df_view_less.empty:
                view_select_less = df_view_less.iloc[df_view_less['view_anchors_avg_norm'].argmax()].name
            else:
                view_select_less = df_view.iloc[df_view['view_anchors_avg_norm'].argmin()].name
            view_select_list = [view_select_less, view_select_larger]
            pdb.set_trace()
            # view_select_list = df_view.iloc[(df_view['view_anchors_avg_norm'] - view_coord).abs().argsort()[:2]].index.tolist()
            df_view_select0 = df.loc[df['anchor_view_name'] == view_select_list[0]]
            df_view_select1 = df.loc[df['anchor_view_name'] == view_select_list[1]]

            # in each view, select two expressions bounding the given exp_coord
            df_view_select0_exp_larger = df_view_select0.loc[(df_view_select0['anchor_pc1_norm'] - exp_coord) > 0]
            if not df_view_select0_exp_larger.empty:
                view_select0_exp_larger = df_view_select0_exp_larger.iloc[df_view_select0_exp_larger['anchor_pc1_norm'].argmin()].name
            else:
                view_select0_exp_larger = df_view_select0.iloc[df_view_select0['anchor_pc1_norm'].argmax()].name
            df_view_select0_exp_less = df_view_select0.loc[(df_view_select0['anchor_pc1_norm'] - exp_coord) <= 0]
            if not df_view_select0_exp_less.empty:
                view_select0_exp_less = df_view_select0_exp_less.iloc[df_view_select0_exp_less['anchor_pc1_norm'].argmax()].name
            else:
                view_select0_exp_less = df_view_select0.iloc[df_view_select0['anchor_pc1_norm'].argmin()].name
            view_select0_exp_list = [view_select0_exp_less, view_select0_exp_larger]
            # view_select0_exp_list = df_view_select0.iloc[(df_view_select0['anchor_pc1_norm'] - exp_coord).abs().argsort()[:2]].index.tolist()
            df_view_select1_exp_larger = df_view_select1.loc[(df_view_select1['anchor_pc1_norm'] - exp_coord) > 0]
            if not df_view_select1_exp_larger.empty:
                view_select1_exp_larger = df_view_select1_exp_larger.iloc[df_view_select1_exp_larger['anchor_pc1_norm'].argmin()].name
            else:
                view_select1_exp_larger = df_view_select1.iloc[df_view_select1['anchor_pc1_norm'].argmax()].name
            df_view_select1_exp_less = df_view_select1.loc[(df_view_select1['anchor_pc1_norm'] - exp_coord) <= 0]
            if not df_view_select1_exp_less.empty:
                view_select1_exp_less = df_view_select1_exp_less.iloc[df_view_select1_exp_less['anchor_pc1_norm'].argmax()].name
            else:
                view_select1_exp_less = df_view_select1.iloc[df_view_select1['anchor_pc1_norm'].argmin()].name
            view_select1_exp_list = [view_select1_exp_less, view_select1_exp_larger]
            # view_select1_exp_list = df_view_select1.iloc[(df_view_select1['anchor_pc1_norm'] - exp_coord).abs().argsort()[:2]].index.tolist()

            # interpolate the four bounding anchors to generate selected latent code
            anchor_select_list = view_select0_exp_list + view_select1_exp_list
            search_point = np.array([view_coord, exp_coord])
            anchor_select_pca_dict = {}
            dist_sum = 0
            for ve in anchor_select_list:
                ve_coord = np.array([df.loc[ve].anchor_pc0_norm, df.loc[ve].anchor_pc1_norm])
                ve_dist = np.linalg.norm(ve_coord - search_point)
                dist_sum += ve_dist
                anchor_select_pca_dict[ve] = {'coord':ve_coord, 'dist': ve_dist}
            dist_weight_sum = 0
            for ve in anchor_select_list:
                anchor_select_pca_dict[ve]['dist_weight'] = dist_sum / anchor_select_pca_dict[ve]['dist']
                dist_weight_sum += anchor_select_pca_dict[ve]['dist_weight']
            for ve in anchor_select_list:
                anchor_select_pca_dict[ve]['dist_weight_norm'] = anchor_select_pca_dict[ve]['dist_weight'] / dist_weight_sum
            latent_select = 0
            # latent should be N*18*512
            for ve in anchor_select_list:
                view_name, exp_name = ve.split('_')
                anchor_ve = anchors_frame_dict_dict[view_name][exp_name]
                anchor_ve_weight = anchor_select_pca_dict[ve]['dist_weight_norm']
                latent_select +=  anchor_ve * anchor_ve_weight
                
            # pdb.set_trace()
            img = generator(latent_select, noise_mode='const', force_fp32=True)[0]
            output_path.joinpath('coord_select').mkdir(exist_ok=True, parents=True)
            output_path.joinpath('coord_select', 'images').mkdir(exist_ok=True, parents=True)
            output_path.joinpath('coord_select', 'latents').mkdir(exist_ok=True, parents=True)    
            save_image(img, output_path.joinpath('coord_select', 'images', 'img_v{:.4f}_e{:.4f}.jpg'.format(view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
            io_utils.save_latents(latent_select, output_path.joinpath('coord_select', 'latents', 'latent_v{:.4f}_e{:.4f}.pt'.format(view_coord, exp_coord)))


def synthesize_fb_multifaces_pca_average(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    '''
    average anchors along principal components.
    and then generate latents based on averaged latent basis given view_exp coordinates.
    '''
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    import pandas as pd
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    U, S, V = torch.pca_lowrank(X)
    # low-dimensional reconstruction
    rd = 2 # number of reduced dimensions to use
    X_proj = torch.matmul(X, V[:, :rd])
    anchors_proj = X_proj.cpu().numpy()
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    view_names, view_anchors_avgs, view_anchors_avg_norms = [], [], []
    for view_key in anchors_frame_dict.keys():
        anchors_view = anchors_frame_dict[view_key]
        anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
        anchors_view_proj = torch.matmul(anchors_view, V[:, :rd]).cpu().numpy()
        anchor_pc0 = anchors_view_proj[:, 0].mean()
        anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
        view_names.append(view_key)
        view_anchors_avgs.append(anchor_pc0)
        view_anchors_avg_norms.append(anchor_pc0_norm)
    data_view = {
        'view_anchors_avg': view_anchors_avgs, \
        'view_anchors_avg_norm': view_anchors_avg_norms, \
        }
    df_view = pd.DataFrame(data_view, index=view_names)

    exp_names, exp_anchors_avgs, exp_anchors_avg_norms = [], [], []
    for exp_key in anchors_exp_dict.keys():
        anchors_exp = anchors_exp_dict[exp_key]
        anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
        anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd]).cpu().numpy()
        anchor_pc1 = anchors_exp_proj[:, 1].mean()
        anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
        exp_names.append(exp_key)
        exp_anchors_avgs.append(anchor_pc1)
        exp_anchors_avg_norms.append(anchor_pc1_norm)
    data_exp = {
        'exp_anchors_avg': exp_anchors_avgs, \
        'exp_anchors_avg_norm': exp_anchors_avg_norms, \
        }
    df_exp = pd.DataFrame(data_exp, index=exp_names)

    anchor_names, anchor_pc0s, anchor_pc1s, anchor_pc0_norms, anchor_pc1_norms = [], [], [], [], []
    anchor_view_names, anchor_exp_names = [], []
    for view_key in anchors_frame_dict_dict.keys():
        for exp_key in anchors_frame_dict_dict[view_key].keys():
            anchor_name = view_key + '_' + exp_key
            anchor = anchors_frame_dict_dict[view_key][exp_key]
            anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd]).cpu().numpy()
            anchor_pc0 = anchor_proj[0][0]
            anchor_pc1 = anchor_proj[0][1]
            anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
            anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
            anchor_names.append(anchor_name)
            anchor_pc0s.append(anchor_pc0)
            anchor_pc1s.append(anchor_pc1)
            anchor_pc0_norms.append(anchor_pc0_norm)
            anchor_pc1_norms.append(anchor_pc1_norm)
            anchor_view_names.append(view_key)
            anchor_exp_names.append(exp_key)
    data = {
        'anchor_view_name': anchor_view_names, \
        'anchor_exp_name': anchor_exp_names, \
        'anchor_pc0': anchor_pc0s, \
        'anchor_pc1': anchor_pc1s, \
        'anchor_pc0_norm': anchor_pc0_norms, \
        'anchor_pc1_norm': anchor_pc1_norms
        }
    df = pd.DataFrame(data, index=anchor_names)
    
    # generate expression average image
    output_path.joinpath('coord_select_strategy2_average').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_exp').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_exp', 'images').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_exp', 'latents').mkdir(exist_ok=True, parents=True)
    for ek in anchors_exp_dict.keys():
        latent_select = anchors_exp_dict[ek].mean(dim=0)
        img = generator(latent_select, noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_exp', 'images', 'exp_avg_{:.4f}_{}.jpg'.format(df_exp.loc[ek].exp_anchors_avg_norm, ek)), nrow=1, normalize=True, range=(-1, 1))
        io_utils.save_latents(latent_select, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_exp', 'latents', 'exp_avg_{:.4f}_{}.pt'.format(df_exp.loc[ek].exp_anchors_avg_norm, ek)))

    # generate view average image
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_view').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_view', 'images').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_view', 'latents').mkdir(exist_ok=True, parents=True)
    for vk in anchors_frame_dict.keys():
        latent_select = anchors_frame_dict[vk].mean(dim=0)
        img = generator(latent_select, noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_view', 'images', 'view_avg_{:.4f}_{}.jpg'.format(df_view.loc[vk].view_anchors_avg_norm, vk)), nrow=1, normalize=True, range=(-1, 1))
        io_utils.save_latents(latent_select, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg_view', 'latents', 'view_avg_{:.4f}_{}.pt'.format(df_view.loc[vk].view_anchors_avg_norm, vk)))

    pdb.set_trace()
    
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg', 'images').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg', 'latents').mkdir(exist_ok=True, parents=True)    

    view_num, exp_num = 50, 50
    for v in range(view_num):
        for e in range(exp_num):
            view_coord, exp_coord = np.clip(1 / view_num * v, 0.01, 0.99), np.clip(1 / exp_num * e, 0.01, 0.99)
            # view_coord, exp_coord = 0.3, 0.6

            # select two views bounding the given view_coord
            df_view_larger = df_view.loc[(df_view['view_anchors_avg_norm'] - view_coord) > 0]
            if not df_view_larger.empty:
                view_select_larger = df_view_larger.iloc[df_view_larger['view_anchors_avg_norm'].argmin()].name
            else:
                view_select_larger = df_view.iloc[df_view['view_anchors_avg_norm'].argmax()].name
            df_view_less = df_view.loc[(df_view['view_anchors_avg_norm'] - view_coord) <= 0]
            if not df_view_less.empty:
                view_select_less = df_view_less.iloc[df_view_less['view_anchors_avg_norm'].argmax()].name
            else:
                view_select_less = df_view.iloc[df_view['view_anchors_avg_norm'].argmin()].name
            view_select_list = [view_select_less, view_select_larger]
            view_lower_ratio = (view_coord - df_view.loc[view_select_list[0]].view_anchors_avg_norm) / \
                (df_view.loc[view_select_list[1]].view_anchors_avg_norm - df_view.loc[view_select_list[0]].view_anchors_avg_norm)
            view_upper_ratio = (df_view.loc[view_select_list[1]].view_anchors_avg_norm - view_coord) / \
                (df_view.loc[view_select_list[1]].view_anchors_avg_norm - df_view.loc[view_select_list[0]].view_anchors_avg_norm)
            latent_select_view = \
                anchors_frame_dict[view_select_list[0]].mean(dim=0) * view_lower_ratio + \
                anchors_frame_dict[view_select_list[1]].mean(dim=0) * view_upper_ratio
                
            # select two expression bounding the given exp_coord
            df_exp_larger = df_exp.loc[(df_exp['exp_anchors_avg_norm'] - exp_coord) > 0]
            if not df_exp_larger.empty:
                exp_select_larger = df_exp_larger.iloc[df_exp_larger['exp_anchors_avg_norm'].argmin()].name
            else:
                exp_select_larger = df_exp.iloc[df_exp['exp_anchors_avg_norm'].argmax()].name
            df_exp_less = df_exp.loc[(df_exp['exp_anchors_avg_norm'] - exp_coord) <= 0]
            if not df_exp_less.empty:
                exp_select_less = df_exp_less.iloc[df_exp_less['exp_anchors_avg_norm'].argmax()].name
            else:
                exp_select_less = df_exp.iloc[df_exp['exp_anchors_avg_norm'].argmin()].name
            exp_select_list = [exp_select_less, exp_select_larger]
            exp_lower_ratio = (exp_coord - df_exp.loc[exp_select_list[0]].exp_anchors_avg_norm) / \
                (df_exp.loc[exp_select_list[1]].exp_anchors_avg_norm - df_exp.loc[exp_select_list[0]].exp_anchors_avg_norm)
            exp_upper_ratio = (df_exp.loc[exp_select_list[1]].exp_anchors_avg_norm - exp_coord) / \
                (df_exp.loc[exp_select_list[1]].exp_anchors_avg_norm - df_exp.loc[exp_select_list[0]].exp_anchors_avg_norm)
            latent_select_exp = \
                anchors_exp_dict[exp_select_list[0]].mean(dim=0) * exp_lower_ratio + \
                anchors_exp_dict[exp_select_list[1]].mean(dim=0) * exp_upper_ratio
            
            latent_select = (latent_select_view + latent_select_exp) / 2

            # pdb.set_trace()
            # latent should be N*18*512
            img = generator(latent_select, noise_mode='const', force_fp32=True)[0]
            save_image(img, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg', 'images', 'img_v{:.4f}_e{:.4f}.jpg'.format(view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
            io_utils.save_latents(latent_select, output_path.joinpath('coord_select_strategy2_average', 'coord_select_avg', 'latents', 'latent_v{:.4f}_e{:.4f}.pt'.format(view_coord, exp_coord)))


def synthesize_fb_multifaces_pca_knn4(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    import pandas as pd
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    U, S, V = torch.pca_lowrank(X)
    # low-dimensional reconstruction
    rd = 2 # number of reduced dimensions to use
    X_proj = torch.matmul(X, V[:, :rd])
    anchors_proj = X_proj.cpu().numpy()
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    view_names, view_anchors_avgs, view_anchors_avg_norms = [], [], []
    for view_key in anchors_frame_dict.keys():
        anchors_view = anchors_frame_dict[view_key]
        anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
        anchors_view_proj = torch.matmul(anchors_view, V[:, :rd]).cpu().numpy()
        anchor_pc0 = anchors_view_proj[:, 0].mean()
        anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
        view_names.append(view_key)
        view_anchors_avgs.append(anchor_pc0)
        view_anchors_avg_norms.append(anchor_pc0_norm)
    data_view = {
        'view_anchors_avg': view_anchors_avgs, \
        'view_anchors_avg_norm': view_anchors_avg_norms, \
        }
    df_view = pd.DataFrame(data_view, index=view_names)

    exp_names, exp_anchors_avgs, exp_anchors_avg_norms = [], [], []
    for exp_key in anchors_exp_dict.keys():
        anchors_exp = anchors_exp_dict[exp_key]
        anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
        anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd]).cpu().numpy()
        anchor_pc1 = anchors_exp_proj[:, 1].mean()
        anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
        exp_names.append(exp_key)
        exp_anchors_avgs.append(anchor_pc1)
        exp_anchors_avg_norms.append(anchor_pc1_norm)
    data_exp = {
        'exp_anchors_avg': exp_anchors_avgs, \
        'exp_anchors_avg_norm': exp_anchors_avg_norms, \
        }
    df_exp = pd.DataFrame(data_exp, index=exp_names)

    anchor_names, anchor_pc0s, anchor_pc1s, anchor_pc0_norms, anchor_pc1_norms = [], [], [], [], []
    anchor_view_names, anchor_exp_names = [], []
    for view_key in anchors_frame_dict_dict.keys():
        for exp_key in anchors_frame_dict_dict[view_key].keys():
            anchor_name = view_key + '_' + exp_key
            anchor = anchors_frame_dict_dict[view_key][exp_key]
            anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd]).cpu().numpy()
            anchor_pc0 = anchor_proj[0][0]
            anchor_pc1 = anchor_proj[0][1]
            anchor_pc0_norm = (anchor_pc0 - pc0_min) / (pc0_max - pc0_min)
            anchor_pc1_norm = (anchor_pc1 - pc1_min) / (pc1_max - pc1_min)
            anchor_names.append(anchor_name)
            anchor_pc0s.append(anchor_pc0)
            anchor_pc1s.append(anchor_pc1)
            anchor_pc0_norms.append(anchor_pc0_norm)
            anchor_pc1_norms.append(anchor_pc1_norm)
            anchor_view_names.append(view_key)
            anchor_exp_names.append(exp_key)
    data = {
        'anchor_view_name': anchor_view_names, \
        'anchor_exp_name': anchor_exp_names, \
        'anchor_pc0': anchor_pc0s, \
        'anchor_pc1': anchor_pc1s, \
        'anchor_pc0_norm': anchor_pc0_norms, \
        'anchor_pc1_norm': anchor_pc1_norms
        }
    df = pd.DataFrame(data, index=anchor_names)

    output_path.joinpath('coord_select_strategy3_knn4').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy3_knn4', 'images').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy3_knn4', 'latents').mkdir(exist_ok=True, parents=True)    

    view_num, exp_num = 50, 50
    for v in range(view_num):
        for e in range(exp_num):
            view_coord, exp_coord = np.clip(1 / view_num * v, 0.01, 0.99), np.clip(1 / exp_num * e, 0.01, 0.99)
            # view_coord, exp_coord = 0.3, 0.6

            anchor_select_list = df.iloc[((df['anchor_pc0_norm'] - view_coord) ** 2 + (df['anchor_pc1_norm'] - exp_coord) ** 2).argsort()[:4]].index.tolist()

            # interpolate the four bounding anchors to generate selected latent code
            search_point = np.array([view_coord, exp_coord])
            anchor_select_pca_dict = {}
            dist_sum = 0
            for ve in anchor_select_list:
                ve_coord = np.array([df.loc[ve].anchor_pc0_norm, df.loc[ve].anchor_pc1_norm])
                ve_dist = np.linalg.norm(ve_coord - search_point)
                dist_sum += ve_dist
                anchor_select_pca_dict[ve] = {'coord':ve_coord, 'dist': ve_dist}
            dist_weight_sum = 0
            for ve in anchor_select_list:
                anchor_select_pca_dict[ve]['dist_weight'] = dist_sum / anchor_select_pca_dict[ve]['dist']
                dist_weight_sum += anchor_select_pca_dict[ve]['dist_weight']
            for ve in anchor_select_list:
                anchor_select_pca_dict[ve]['dist_weight_norm'] = anchor_select_pca_dict[ve]['dist_weight'] / dist_weight_sum
            latent_select = 0
            # latent should be N*18*512
            for ve in anchor_select_list:
                view_name, exp_name = ve.split('_')
                anchor_ve = anchors_frame_dict_dict[view_name][exp_name]
                anchor_ve_weight = anchor_select_pca_dict[ve]['dist_weight_norm']
                latent_select +=  anchor_ve * anchor_ve_weight
                
            # pdb.set_trace()
            img = generator(latent_select, noise_mode='const', force_fp32=True)[0]
            save_image(img, output_path.joinpath('coord_select_strategy3_knn4', 'images', 'img_v{:.4f}_e{:.4f}.jpg'.format(view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
            io_utils.save_latents(latent_select, output_path.joinpath('coord_select_strategy3_knn4', 'latents', 'latent_v{:.4f}_e{:.4f}.pt'.format(view_coord, exp_coord)))


def synthesize_fb_multifaces_pca_pc(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    import pandas as pd
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    import time
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    # U, S, V = torch.pca_lowrank(A=X, q=X.shape[0])
    
    ## PCA timing checking
    # for q in range(anchors_all.shape[0]):
    #     pca_st_time = time.time()
    #     U, S, V = torch.pca_lowrank(A=X, q=q)
    #     pca_ed_time = time.time()
    #     print('pca {} timing:{}'.format(q, pca_ed_time - pca_st_time))

    # # low-dimensional reconstruction
    # rd = 2 # number of reduced dimensions to use
    # X_proj = torch.matmul(X, V)
    # anchors_proj = X_proj.cpu().numpy()
    # pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    # pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    # X_inv = torch.matmul(X_proj, V.T)
    # X_inv = torch.matmul(V, X_proj)
    # latent = X.reshape(anchors_all.shape).squeeze()
    # latent_inv = X_inv.reshape(anchors_all.shape).squeeze()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99)
    anchors_proj = pca.fit_transform(X.cpu().numpy())
    latent = X.reshape(anchors_all.shape).squeeze()
    latent_inv  = torch.from_numpy(pca.inverse_transform(anchors_proj).reshape(anchors_all.shape).squeeze()).cuda()
    pdb.set_trace()
    
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    output_path.joinpath('coord_select_strategy4_pc').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc', 'images').mkdir(exist_ok=True, parents=True)

    is_average_base = False
    is_lock_down_v_e_for_all = False
    if is_average_base:
        output_path.joinpath('coord_select_strategy4_pc', 'images_average').mkdir(exist_ok=True, parents=True)
        # pdb.set_trace()
        anchors_proj_cp = np.copy(anchors_proj.mean(0)[None, :])
        view_num, exp_num = 25, 25
        for v in range(view_num):
            for e in range(exp_num):
                view_coord = np.clip(1 / view_num * v, 0.01, 0.99)
                v_pc = view_coord * (pc0_max - pc0_min) + pc0_min
                exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                e_pc = exp_coord * (pc1_max - pc1_min) + pc1_min
                anchors_proj_cp[0][0] = v_pc
                anchors_proj_cp[0][1] = e_pc
                latent_inv  = torch.from_numpy(pca.inverse_transform(anchors_proj_cp).reshape(anchors_all.shape[1:]).squeeze()).cuda()
                img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'images_average', 'img_v_{:4f}_e_{:4f}.jpg'.format(view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
    else:
        if is_lock_down_v_e_for_all:
            output_path.joinpath('coord_select_strategy4_pc', 'images_v_e_fixed').mkdir(exist_ok=True, parents=True)
            view_num, exp_num = 5, 5
            for v in range(view_num):
                for e in range(exp_num):
                    view_coord = np.clip(1 / view_num * v, 0.01, 0.99)
                    v_pc = view_coord * (pc0_max - pc0_min) + pc0_min
                    exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                    e_pc = exp_coord * (pc1_max - pc1_min) + pc1_min
                    for kv in anchors_frame_dict_dict.keys():
                        for ke in anchors_frame_dict_dict[kv].keys():
                            anchor = anchors_frame_dict_dict[kv][ke].reshape(1, -1).cpu().numpy()
                            anchor_proj = pca.transform(anchor)
                            anchor_proj_cp = np.copy(anchor_proj)
                            anchor_proj_cp[0][0] = v_pc
                            anchor_proj_cp[0][1] = e_pc
                            latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_cp).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                            img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                            save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'images_v_e_fixed', 'img_{}_{}_v{:4f}_e{:4f}.jpg'.format(kv, ke, view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
        else:
            for kv in anchors_frame_dict_dict.keys():
                for ke in anchors_frame_dict_dict[kv].keys():
                    anchor = anchors_frame_dict_dict[kv][ke].reshape(1, -1).cpu().numpy()
                    anchor_proj = pca.transform(anchor)
                    anchor_proj_view = np.copy(anchor_proj)
                    anchor_proj_exp = np.copy(anchor_proj)
                    anchor_proj_both = np.copy(anchor_proj)
                    view_num, exp_num = 25, 25
                    for v in range(view_num):
                        view_coord = np.clip(1 / view_num * v, 0.01, 0.99)
                        v_pc = view_coord * (pc0_max - pc0_min) + pc0_min
                        anchor_proj_view[0][0] = v_pc
                        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_view).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                        img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'images', 'img_{}_{}_v{:4f}.jpg'.format(kv, ke, view_coord)), nrow=1, normalize=True, range=(-1, 1))
                        anchor_proj_both[0][0] = v_pc
                        anchor_proj_both[0][1] = v_pc
                        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_both).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                        img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'images', 'img_{}_{}_ve{:4f}.jpg'.format(kv, ke, view_coord)), nrow=1, normalize=True, range=(-1, 1))
                    for e in range(exp_num):
                        exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                        e_pc = exp_coord * (pc1_max - pc1_min) + pc1_min
                        anchor_proj_exp[0][1] = e_pc
                        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_exp).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                        img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'images', 'img_{}_{}_e_{:4f}.jpg'.format(kv, ke, exp_coord)), nrow=1, normalize=True, range=(-1, 1))

    
    pdb.set_trace()


    import matplotlib.pyplot as plt
    color_list = ['yellow', 'black', 'green', 'orange', 'navy', 'pink', 'purple', 'red', \
                'sienna', 'salmon', 'dimgray', 'silver', 'springgreen', 'aquamarine', 'teal', 'cyan', \
                'violet', 'blue', 'cornflowerblue', 'plum', 'purple', 'palevioletred', 'greenyellow', 'wheat', \
                ]
    output_path.joinpath('coord_select_strategy4_pc').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc', 'recon_plots').mkdir(exist_ok=True, parents=True)
    cnt = 0
    scatter_x, scatter_y = [], []
    for i in range(len(anchors_proj)): 
        scatter_x.append((anchors_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
        scatter_y.append((anchors_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
    plt.scatter(scatter_x, scatter_y, color='black')
    plt.xlabel('PCA C1')
    plt.ylabel('PCA C2')
    plt.title('PCA Projection of Anchors')
    plt.savefig(output_path.joinpath('coord_select_strategy4_pc', 'recon_plots', 'anchors_pca2_plots.png'))

    
    pdb.set_trace()
    output_path.joinpath('coord_select_strategy4_pc').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc', 'recon').mkdir(exist_ok=True, parents=True)

    for i in range(X.shape[0]):
        img = generator(latent[i].unsqueeze(0), noise_mode='const', force_fp32=True)[0]
        img_inv = generator(latent_inv[i].unsqueeze(0), noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('coord_select_strategy4_pc', 'recon', 'img_{}.jpg'.format(str(i).zfill(3))), nrow=1, normalize=True, range=(-1, 1))
        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'recon', 'img_{}_inv.jpg'.format(str(i).zfill(3))), nrow=1, normalize=True, range=(-1, 1))
        # pdb.set_trace()


def synthesize_fb_multifaces_pca_pc_vyvp(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    generator = io_utils.load_net(generator_path).to('cuda')
    # import pdb; pdb.set_trace()
    import pandas as pd
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_coord_access_vyvp(anchors_path)
    
    import time
    X = anchors_all.reshape(anchors_all.shape[0], -1)
    # U, S, V = torch.pca_lowrank(A=X, q=X.shape[0])
    
    ## PCA timing checking
    # for q in range(anchors_all.shape[0]):
    #     pca_st_time = time.time()
    #     U, S, V = torch.pca_lowrank(A=X, q=q)
    #     pca_ed_time = time.time()
    #     print('pca {} timing:{}'.format(q, pca_ed_time - pca_st_time))

    # # low-dimensional reconstruction
    # rd = 2 # number of reduced dimensions to use
    # X_proj = torch.matmul(X, V)
    # anchors_proj = X_proj.cpu().numpy()
    # pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    # pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()

    # X_inv = torch.matmul(X_proj, V.T)
    # X_inv = torch.matmul(V, X_proj)
    # latent = X.reshape(anchors_all.shape).squeeze()
    # latent_inv = X_inv.reshape(anchors_all.shape).squeeze()
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99)
    anchors_proj = pca.fit_transform(X.cpu().numpy())
    latent = X.reshape(anchors_all.shape).squeeze()
    latent_inv  = torch.from_numpy(pca.inverse_transform(anchors_proj).reshape(anchors_all.shape).squeeze()).cuda()
    
    pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
    pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()
    pc2_min, pc2_max = anchors_proj[:, 2].min(), anchors_proj[:, 2].max()

    output_path.joinpath('coord_select_strategy4_pc_vyvp').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images').mkdir(exist_ok=True, parents=True)

    is_average_base = False
    is_lock_down_v_e_for_all = False
    if is_average_base:
        output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images_average').mkdir(exist_ok=True, parents=True)
        # pdb.set_trace()
        anchors_proj_cp = np.copy(anchors_proj.mean(0)[None, :])
        view_num, exp_num = 50, 50
        for v in range(view_num):
            for e in range(exp_num):
                view_coord = np.clip(1 / view_num * v, 0.01, 0.99)
                v_pc = view_coord * (pc0_max - pc0_min) + pc0_min
                exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                e_pc = exp_coord * (pc1_max - pc1_min) + pc1_min
                anchors_proj_cp[0][0] = v_pc
                anchors_proj_cp[0][1] = e_pc
                latent_inv  = torch.from_numpy(pca.inverse_transform(anchors_proj_cp).reshape(anchors_all.shape[1:]).squeeze()).cuda()
                img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images_average', 'img_v_{:4f}_e_{:4f}.jpg'.format(view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
    else:
        if is_lock_down_v_e_for_all:
            output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images_v_e_fixed').mkdir(exist_ok=True, parents=True)
            view_num, exp_num = 5, 5
            for v in range(view_num):
                for e in range(exp_num):
                    view_coord = np.clip(1 / view_num * v, 0.01, 0.99)
                    v_pc = view_coord * (pc0_max - pc0_min) + pc0_min
                    exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                    e_pc = exp_coord * (pc1_max - pc1_min) + pc1_min
                    for kv in anchors_frame_dict_dict.keys():
                        for ke in anchors_frame_dict_dict[kv].keys():
                            anchor = anchors_frame_dict_dict[kv][ke].reshape(1, -1).cpu().numpy()
                            anchor_proj = pca.transform(anchor)
                            anchor_proj_cp = np.copy(anchor_proj)
                            anchor_proj_cp[0][0] = v_pc
                            anchor_proj_cp[0][1] = e_pc
                            latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_cp).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                            img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                            save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images_v_e_fixed', 'img_{}_{}_v{:4f}_e{:4f}.jpg'.format(kv, ke, view_coord, exp_coord)), nrow=1, normalize=True, range=(-1, 1))
        else:
            for kv in anchors_frame_dict_dict.keys():
                for ke in anchors_frame_dict_dict[kv].keys():
                    anchor = anchors_frame_dict_dict[kv][ke].reshape(1, -1).cpu().numpy()
                    anchor_proj = pca.transform(anchor)
                    anchor_proj_view = np.copy(anchor_proj)
                    anchor_proj_exp = np.copy(anchor_proj)
                    viewy_num, viewp_num, exp_num = 10, 10, 50
                    for vy in range(viewy_num):
                        viewy_coord = np.clip(1 / viewy_num * vy, 0.01, 0.99)
                        vy_pc = viewy_coord * (pc0_max - pc0_min) + pc0_min
                        anchor_proj_view[0][0] = vy_pc
                        for vp in range(viewp_num):
                            viewp_coord = np.clip(1 / viewp_num * vp, 0.01, 0.99)
                            vp_pc = viewp_coord * (pc1_max - pc1_min) + pc1_min
                            anchor_proj_view[0][1] = vp_pc
                            latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_view).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                            img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                            save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images', 'img_{}_{}_vy{:4f}_vp{:4f}.jpg'.format(kv, ke, viewy_coord, viewp_coord)), nrow=1, normalize=True, range=(-1, 1))
                    for e in range(exp_num):
                        exp_coord = np.clip(1 / exp_num * e, 0.01, 0.99)
                        e_pc = exp_coord * (pc2_max - pc2_min) + pc2_min
                        anchor_proj_exp[0][2] = e_pc
                        latent_inv  = torch.from_numpy(pca.inverse_transform(anchor_proj_exp).reshape(anchors_frame_dict_dict[kv][ke].shape).squeeze()).cuda()
                        img_inv = generator(latent_inv.unsqueeze(0), noise_mode='const', force_fp32=True)[0]
                        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc_vyvp', 'images', 'img_{}_{}_e_{:4f}.jpg'.format(kv, ke, exp_coord)), nrow=1, normalize=True, range=(-1, 1))

    
    pdb.set_trace()


    import matplotlib.pyplot as plt
    color_list = ['yellow', 'black', 'green', 'orange', 'navy', 'pink', 'purple', 'red', \
                'sienna', 'salmon', 'dimgray', 'silver', 'springgreen', 'aquamarine', 'teal', 'cyan', \
                'violet', 'blue', 'cornflowerblue', 'plum', 'purple', 'palevioletred', 'greenyellow', 'wheat', \
                ]
    output_path.joinpath('coord_select_strategy4_pc').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc', 'recon_plots').mkdir(exist_ok=True, parents=True)
    cnt = 0
    scatter_x, scatter_y = [], []
    for i in range(len(anchors_proj)): 
        scatter_x.append((anchors_proj[i][0] - pc0_min) / (pc0_max - pc0_min))
        scatter_y.append((anchors_proj[i][1] - pc1_min) / (pc1_max - pc1_min))
    plt.scatter(scatter_x, scatter_y, color='black')
    plt.xlabel('PCA C1')
    plt.ylabel('PCA C2')
    plt.title('PCA Projection of Anchors')
    plt.savefig(output_path.joinpath('coord_select_strategy4_pc', 'recon_plots', 'anchors_pca2_plots.png'))

    
    pdb.set_trace()
    output_path.joinpath('coord_select_strategy4_pc').mkdir(exist_ok=True, parents=True)
    output_path.joinpath('coord_select_strategy4_pc', 'recon').mkdir(exist_ok=True, parents=True)

    for i in range(X.shape[0]):
        img = generator(latent[i].unsqueeze(0), noise_mode='const', force_fp32=True)[0]
        img_inv = generator(latent_inv[i].unsqueeze(0), noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('coord_select_strategy4_pc', 'recon', 'img_{}.jpg'.format(str(i).zfill(3))), nrow=1, normalize=True, range=(-1, 1))
        save_image(img_inv, output_path.joinpath('coord_select_strategy4_pc', 'recon', 'img_{}_inv.jpg'.format(str(i).zfill(3))), nrow=1, normalize=True, range=(-1, 1))
        # pdb.set_trace()

def synthesize_fb_multifaces_view_pair(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    '''
    given two views and multiple sharing expressions of them, 
    synthesize sharing novel expressions of the views.
    '''
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()
    frame_list = ['400060', '400061']    
    exp_list = ['E001', 'E004', 'E005', 'E008', 'E010', 'E016']

    anchors_dict = io_utils.load_latents_fb_multifaces_view_pair(anchors_path, frame_list, exp_list)
    
    pdb.set_trace()
    frame_pair_name = frame_list[0] + '_' + frame_list[1]
    output_path.joinpath(frame_pair_name).mkdir(exist_ok=True, parents=True)
    for frame in frame_list:
        output_path.joinpath(frame_pair_name, frame).mkdir(exist_ok=True, parents=True)
        output_path.joinpath(frame_pair_name, frame, 'images').mkdir(exist_ok=True, parents=True)
        output_path.joinpath(frame_pair_name, frame, 'latents').mkdir(exist_ok=True, parents=True)

    latents_dict = latent_space_ops.sample_from_P0_view_pair(anchors_dict, num_points_to_sample, num_anchor_points)
    for frame in frame_list:
        latents = latents_dict[frame]
        batch_size = 4
        i = 0
        while i < latents.shape[0]:
            lats = latents[i: i + batch_size]
            imgs = generator(lats.squeeze(1), noise_mode='const', force_fp32=True)

            for j in range(min(batch_size, num_points_to_sample - i)):
                save_image(imgs[j], output_path.joinpath(frame_pair_name, frame, 'images', f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))
                io_utils.save_latents(lats[j], output_path.joinpath(frame_pair_name, frame, 'latents', f'{i + j}.pt'))

            del imgs
            i += batch_size

def synthesize_facescape_views_interpolation(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    '''
    given two images of the same view and different expressions on FaceScape dataset, figure 9
    '''
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()

    # # fs, f9
    # exp = 'E10'
    # view_list = ['V010', 'V030']
    
    # fb, f0
    exp = 'E013' # 'E010' #'E001'
    view_list = ['400037', '400039'] #['400060', '400061']
    ratio_list = [0.2*i for i in range(6)]
    
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    latent0 = anchors_exp_dict_dict[exp][view_list[0]]
    latent1 = anchors_exp_dict_dict[exp][view_list[1]]
    
    output_path.joinpath('images_interpolation').mkdir(exist_ok=True, parents=True)
    for ratio in ratio_list:
        latent = latent0 * ratio + latent1 * (1 - ratio)
        img_int_name = '{}_interpolation_{}_{}_ratio_{:4f}.jpg'.format(exp, view_list[0], view_list[1], ratio)
        img = generator(latent, noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('images_interpolation', img_int_name), nrow=1, normalize=True, range=(-1, 1))

def synthesize_facescape_expressions_interpolation(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    '''
    given two images of the same expression and different views on FaceScape dataset, figure 2
    '''
    generator = io_utils.load_net(generator_path).to('cuda')
    import pdb; pdb.set_trace()

    # exp_list = ['E13', 'E15']
    # view = 'V006'

    # exp_list = ['E8', 'E9']
    # view = 'V016'

    exp_list = ['E6', 'E18']
    view = 'V016'

    ratio_list = [0.2*i for i in range(6)]
    
    anchors_all, anchors_exp_dict, anchors_frame_dict, anchors_exp_dict_dict, anchors_frame_dict_dict = \
        io_utils.load_latents_fb_multifaces_coord_access(anchors_path)
    
    latent0 = anchors_frame_dict_dict[view][exp_list[0]]
    latent1 = anchors_frame_dict_dict[view][exp_list[1]]
    
    output_path.joinpath('images_interpolation').mkdir(exist_ok=True, parents=True)
    for ratio in ratio_list:
        latent = latent0 * ratio + latent1 * (1 - ratio)
        img_int_name = '{}_interpolation_{}_{}_ratio_{:4f}.jpg'.format(view, exp_list[0], exp_list[1], ratio)
        img = generator(latent, noise_mode='const', force_fp32=True)[0]
        save_image(img, output_path.joinpath('images_interpolation', img_int_name), nrow=1, normalize=True, range=(-1, 1))


def parse_args(raw_args):
    parser = ArgumentParser()
    parser.add_argument('--anchors_path', required=True, type=Path)
    parser.add_argument('--generator_path', required=True, type=Path)
    parser.add_argument('--output_path', required=True, type=Path)

    parser.add_argument('--device', default='0')

    parser.add_argument('--num_points_to_sample', type=int, default=30)
    parser.add_argument('--num_anchors_for_sampling', type=int, default=3)

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.output_path.mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('latents').mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('images').mkdir(exist_ok=True, parents=True)

    return args


def main(raw_args=None):
    # TODO(2): support W synthesis
    args = parse_args(raw_args)
    args = process_args(args)

    synthesize(args.anchors_path, args.generator_path, args.output_path,
               args.num_points_to_sample, args.num_anchors_for_sampling)

if __name__ == '__main__':
    with torch.no_grad():
        main()
