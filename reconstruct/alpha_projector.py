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

import hyperparams
from reconstruct.base_projector import BaseProjector

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import io_utils

class AlphaProjector(BaseProjector):
    def __init__(self, device, generator, debug_out_path, is_wplus, anchors, degradation_func=None, beta=None,
                 l2_weight=hyperparams.l2_weight,
                 deltas_layers_to_reg=hyperparams.deltas_layers_to_reg,
                 sum_1_reg_weight=hyperparams.sum_1_reg_weight,
                 deltas_reg_weight=hyperparams.deltas_reg_weight):

        self.anchors = anchors.float().to(device).requires_grad_(False)
        self.beta = beta
        self.is_wplus = is_wplus

        self.deltas_layers_to_reg = deltas_layers_to_reg
        self.sum_1_reg_weight = sum_1_reg_weight
        self.deltas_reg_weight = deltas_reg_weight
        self.num_steps = hyperparams.projection_steps

        self.alpha_opt = None
        self.deltas_opt = None

        super(AlphaProjector, self).__init__(device, generator, hyperparams.projection_steps,
                                             debug_out_path, l2_weight, degradation_func)

    def get_alpha_plus(self):
        alpha_plus = self.alpha_opt.T + self.deltas_opt
        return alpha_plus

    def get_constrained_alpha_plus(self):
        alphas = self.get_alpha_plus()
        if self.beta is not None:
            alphas = alphas + self.beta
            alphas = F.softplus(alphas, beta=100)
            alphas = alphas - self.beta

        alphas = alphas / torch.sum(alphas, dim=-1, keepdim=True)

        return alphas

    def get_latent(self):
        constrained_alpha_plus = self.get_constrained_alpha_plus()
        w_opt = constrained_alpha_plus @ self.anchors
        w_opt = w_opt.unsqueeze(0)

        return w_opt

    def get_optimized(self):
        return self.alpha_opt, self.deltas_opt

    def regularization_loss(self):
        alpha_plus = self.get_alpha_plus()

        delta_reg_loss = self.deltas_reg_weight * torch.sum(
            torch.norm(self.deltas_opt[:self.deltas_layers_to_reg], dim=1))
        sum1_1_reg_loss = self.sum_1_reg_weight * torch.mean((torch.sum(alpha_plus, axis=1) - 1) ** 2)

        reg_loss = delta_reg_loss + sum1_1_reg_loss
        return reg_loss

    def set_optimization(self):
        num_anchors = self.anchors.shape[0]
        self.alpha_opt = torch.ones((num_anchors, 1)) / num_anchors
        self.deltas_opt = torch.zeros((self.generator.num_ws, num_anchors))

        self.alpha_opt = self.alpha_opt.float().to(self.device).requires_grad_(True)
        self.deltas_opt = self.deltas_opt.float().to(self.device)

        optim_params = [self.alpha_opt]

        if self.is_wplus:
            self.deltas_opt.requires_grad_(True)
            optim_params.append(self.deltas_opt)

        # TODO(5): currently not supporting noise optimization.
        # noise_bufs = {name: buf for (name, buf) in self.generator.named_buffers() if 'noise_const' in name}
        # noise_params = list(noise_bufs.values())

        self.optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999), lr=hyperparams.projection_lr)

    def save(self, sample):
        latent = self.get_latent()
        alpha_opt, deltas_opt = self.get_optimized() # lbz
        alpha = self.get_constrained_alpha_plus().unsqueeze(0)
        recon_img = self.generator(latent, noise_mode='const', force_fp32=True)

        sample.set(latent, alpha_opt, deltas_opt, alpha, recon_img)


    def reconstruct_controllable(self, sample, exp_coord=0.5, is_super_res=True):
        '''
        controllable image enhancement.
        '''
        self.set_optimization()

        if not is_super_res:
            # PCA for controllable just for inpainting, not super-resolution
            X = self.anchors.reshape(self.anchors.shape[0], -1)
            U, S, V = torch.pca_lowrank(X)
            # low-dimensional reconstruction
            rd = 2 # number of reduced dimensions to use
            X_red = U[:, :rd] @ (torch.diag(S)[:rd, :rd]) @ V.T[:rd, :]
            X_proj = torch.matmul(X, V[:, :rd])
            anchors_proj = X_proj.cpu().numpy()
            pc0_min, pc0_max = anchors_proj[:, 0].min(), anchors_proj[:, 0].max()
            pc1_min, pc1_max = anchors_proj[:, 1].min(), anchors_proj[:, 1].max()
            def pca_proj_normalization(pc0, pc1):
                pc0_norm = (pc0 - pc0_min) / (pc0_max - pc0_min)
                pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min)
                return pc0_norm, pc1_norm

        # TODO(3): implement early stopping
        for step in tqdm(range(self.num_steps)):
            to_visualize = self.need_visualize(step)

            latent_opt = self.get_latent()
            target = sample.img.cuda()
            
            synth = self.generator(latent_opt, noise_mode='const', force_fp32=True)
            
            recon_loss = self.reconstruction_loss(synth, target, mask=sample.mask)
            regularization_loss = self.regularization_loss()

            if True: #not is_super_res:
                latent_opt_w = latent_opt[:, 0:1, :].reshape(latent_opt.shape[0], -1)
                latent_proj = torch.matmul(latent_opt_w, V[:, :rd])
                pc0_norm, pc1_norm = pca_proj_normalization(latent_proj[0][0], latent_proj[0][1])
                loss_anchor_exp = torch.nn.L1Loss()(torch.Tensor([exp_coord]).cuda(), pc1_norm)
                loss_anchor = loss_anchor_exp
                # if step < 500: # only apply anchor loss after certain step
                #     loss_anchor = 0
                if step % 100 == 0:
                    print('exp_coord:{}, pc1_norm:{}'.format(exp_coord, pc1_norm))
                    # import pdb; pdb.set_trace()
                loss = recon_loss + regularization_loss + loss_anchor
            else:
                loss = recon_loss + regularization_loss
            
            if to_visualize:
                io_utils.save_images(synth.detach().cpu(),
                                     self.debug_out_path.joinpath('{}_exp{}_step_{}'.format(sample.name, int(exp_coord*10), step)))
                print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(loss.item()):<5.2f}')

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        self.save(sample)
        
        return sample




    def reconstruct_controllable_expaxis(self, sample, exp_coord=0.5, exp_axis=1, is_super_res=True):
        '''
        controllable image enhancement.
        '''
        self.set_optimization()

        if not is_super_res:
            # PCA for controllable just for inpainting, not super-resolution
            X = self.anchors.reshape(self.anchors.shape[0], -1)
            U, S, V = torch.pca_lowrank(X)
            # low-dimensional reconstruction
            rd = 3 # number of reduced dimensions to use
            X_red = U[:, :rd] @ (torch.diag(S)[:rd, :rd]) @ V.T[:rd, :]
            X_proj = torch.matmul(X, V[:, :rd])
            anchors_proj = X_proj.cpu().numpy()
            pc1_min, pc1_max = anchors_proj[:, exp_axis].min(), anchors_proj[:, exp_axis].max()
            def pca_proj_normalization(pc1):
                pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min)
                return pc1_norm

        # TODO(3): implement early stopping
        for step in tqdm(range(self.num_steps)):
            to_visualize = self.need_visualize(step)

            latent_opt = self.get_latent()
            target = sample.img.cuda()
            
            synth = self.generator(latent_opt, noise_mode='const', force_fp32=True)
            
            recon_loss = self.reconstruction_loss(synth, target, mask=sample.mask)
            regularization_loss = self.regularization_loss()

            if True: #not is_super_res:
                latent_opt_w = latent_opt[:, 0:1, :].reshape(latent_opt.shape[0], -1)
                latent_proj = torch.matmul(latent_opt_w, V[:, :rd])
                pc1_norm = pca_proj_normalization(latent_proj[0][1])
                loss_anchor_exp = torch.nn.L1Loss()(torch.Tensor([exp_coord]).cuda(), pc1_norm.unsqueeze(0))
                loss_anchor = loss_anchor_exp
                # if step < 500: # only apply anchor loss after certain step
                #     loss_anchor = 0
                if step % 100 == 0:
                    print('exp_coord:{}, pc1_norm:{}'.format(exp_coord, pc1_norm))
                    # import pdb; pdb.set_trace()
                loss = recon_loss + regularization_loss + loss_anchor
            else:
                loss = recon_loss + regularization_loss
            
            if to_visualize:
                io_utils.save_images(synth.detach().cpu(),
                                     self.debug_out_path.joinpath('{}_exp{}_step_{}'.format(sample.name, int(exp_coord*10), step)))
                print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(loss.item()):<5.2f}')

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        self.save(sample)
        
        return sample
