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
from utils import io_utils
from reconstruct.base_reconstructor import BaseReconstructor

import torch
from tqdm import tqdm
import time
import pdb
import pickle
from torch.utils.tensorboard import SummaryWriter


class Tuner(BaseReconstructor):
    def __init__(self, device, generator, debug_out_path=None, l2_weight=hyperparams.l2_weight):
        super().__init__(device, generator, hyperparams.tune_steps, debug_out_path, l2_weight)

    def set_optimization(self):
        self.generator.train()
        for p in self.generator.parameters():
            p.requires_grad = True

        # self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=hyperparams.tune_lr)

    def reconstruct(self, dataset):
        '''
        tune both weights of generator, and anchors.
        '''
        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            # TODO(1): batched training
            for sample in dataset:
                anchor = sample.w_code.cuda()
                target = sample.img.cuda()

                synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                loss = self.reconstruction_loss(synth, target)

                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()

            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')

        return self.generator

    def reconstruct_anchor(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False):        
        anchors_all_dict, anchors_exp_dict, anchors_view_dict = {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            # pdb.set_trace()
            sample_name = sample.name
            exp_name = sample_name.split('_')[4] # new celebrities
            view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]            
            # exp_name = sample_name.split('_')[1]
            # view_name = sample_name.split('_')[-2]
            anchors_all_dict[sample_name] = sample_anchor_param
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]

            if is_lightstage:
                light_name = sample_name.split('_')[-1].split('.')[0]
                if light_name in anchors_light_dict.keys():
                    anchors_light_dict[light_name].append(sample_anchor_param)
                else:
                    anchors_light_dict[light_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)

        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            
            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                rd = 2 # number of reduced dimensions to use
                sample_name = sample.name
                exp_name = sample_name.split('_')[4]
                view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]
                if is_lightstage:
                    rd = 3
                    light_name = sample_name.split('_')[-1]
                    anchors_light = torch.stack(anchors_light_dict[light_name], dim=0)
                    anchors_light = anchors_light.reshape(anchors_light.shape[0], -1)
                    anchors_light_proj = torch.matmul(anchors_light, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[0], anchor_proj[0][0])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[1], anchor_proj[0][1])
                if is_lightstage:
                    loss_anchor_light = torch.nn.L1Loss()(torch.mean(anchors_light_proj, dim=0)[2], anchor_proj[0][2])
                    loss_anchor = loss_anchor_exp + loss_anchor_view + loss_anchor_light
                else:
                    loss_anchor = loss_anchor_exp + loss_anchor_view
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                
                # print('pca {} timing:{}'.format(pca_rank, pca_ed_time - pca_st_time))
                # print('anchor loss {} timing:{}'.format(pca_rank, anchor_loss_ed_time - pca_ed_time))
                # print('recon loss {} timing:{}'.format(pca_rank, recon_loss_ed_time - anchor_loss_ed_time))

                # if cnt == 4:
                #     pdb.set_trace()
                # cnt += 1
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        return self.generator
    
    
    def reconstruct_anchor_age(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):        
        anchors_all_dict, anchors_exp_dict, anchors_age_dict, anchors_view_dict = {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            age_name = sample_name.split('_')[0]
            exp_name = sample_name.split('_')[1]
            view_name = sample_name.split('_')[2]

            anchors_all_dict[sample_name] = sample_anchor_param
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]

            if is_lightstage:
                light_name = sample_name.split('_')[-1].split('.')[0]
                if light_name in anchors_light_dict.keys():
                    anchors_light_dict[light_name].append(sample_anchor_param)
                else:
                    anchors_light_dict[light_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)

        pdb.set_trace()
        writer = SummaryWriter(log_dir='runs/{}_{}_viewexpage'.format(sample_name.split('_')[-3], sample_name.split('_')[-2]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            tot_loss_anchor_yaw, tot_loss_anchor_exp, tot_loss_anchor_age = 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                rd = 3 # number of reduced dimensions to use
                sample_name = sample.name
                age_name = sample_name.split('_')[0]
                exp_name = sample_name.split('_')[1]
                view_name = sample_name.split('_')[2]
                if is_lightstage:
                    rd = 3
                    light_name = sample_name.split('_')[-1]
                    anchors_light = torch.stack(anchors_light_dict[light_name], dim=0)
                    anchors_light = anchors_light.reshape(anchors_light.shape[0], -1)
                    anchors_light_proj = torch.matmul(anchors_light, V[:, :rd])
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[0], anchor_proj[0][0])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[1], anchor_proj[0][1])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[2], anchor_proj[0][2])
                if is_lightstage:
                    loss_anchor_light = torch.nn.L1Loss()(torch.mean(anchors_light_proj, dim=0)[2], anchor_proj[0][2])
                    loss_anchor = loss_anchor_exp + loss_anchor_view + loss_anchor_light
                else:
                    loss_anchor = loss_anchor_age + loss_anchor_exp + loss_anchor_view
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_age += loss_anchor_age.item()
                tot_loss_anchor_yaw += loss_anchor_view.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_yaw', tot_loss_anchor_yaw / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_age', tot_loss_anchor_age / len(dataset), step)

            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        return self.generator

    
    
    
    
    def reconstruct_anchor_yawpitchexp_celebrities1(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):
        '''
        compared to previous version,
        automatically find corresponding directions for 
        each attributes.
        '''       
        print('reconstruct_anchor_yawpitchexp_celebrities1')       
        anchors_all_dict, anchors_exp_dict, anchors_yaw_dict, anchors_pitch_dict = {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # eg. barack_obama_VY-16_VP-1_E8_71_align1500.jpg
            yaw_name = sample_name.split('_')[2]
            pitch_name = sample_name.split('_')[3]
            exp_name = sample_name.split('_')[4]

            anchors_all_dict[sample_name] = sample_anchor_param
            if yaw_name in anchors_yaw_dict.keys():
                anchors_yaw_dict[yaw_name].append(sample_anchor_param)
            else:
                anchors_yaw_dict[yaw_name] = [sample_anchor_param]
            
            if pitch_name in anchors_pitch_dict.keys():
                anchors_pitch_dict[pitch_name].append(sample_anchor_param)
            else:
                anchors_pitch_dict[pitch_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        if resume_epoch == 0:
            with torch.no_grad():
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                yaw_proj_var, pitch_proj_var, exp_proj_var = [], [], []
                rd = pca_rank
                for key in anchors_yaw_dict.keys():
                    anchors_yaw = torch.stack(anchors_yaw_dict[key], dim=0)
                    if anchors_yaw.shape[0] == 1:
                        continue
                    anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                    anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :rd])
                    anchors_yaw_proj_var = torch.var(anchors_yaw_proj, dim=0)
                    yaw_proj_var.append(anchors_yaw_proj_var)
                yaw_proj_var = torch.stack(yaw_proj_var, dim=0)
                yaw_proj_var_norm = torch.sum(yaw_proj_var, dim=0) / S
                yaw_axis = torch.argmin(yaw_proj_var_norm).item()

                for key in anchors_pitch_dict.keys():
                    anchors_pitch = torch.stack(anchors_pitch_dict[key], dim=0)
                    if anchors_pitch.shape[0] == 1:
                        continue
                    anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                    anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :rd])
                    anchors_pitch_proj_var = torch.var(anchors_pitch_proj, dim=0)
                    pitch_proj_var.append(anchors_pitch_proj_var)
                pitch_proj_var = torch.stack(pitch_proj_var, dim=0)
                pitch_proj_var_norm = torch.sum(pitch_proj_var, dim=0) / S
                pitch_proj_var_norm[yaw_axis] = pitch_proj_var_norm.max()
                pitch_axis = torch.argmin(pitch_proj_var_norm).item()

                for key in anchors_exp_dict.keys():
                    anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                    if anchors_exp.shape[0] == 1:
                        continue
                    anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                    anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                    anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                    exp_proj_var.append(anchors_exp_proj_var)
                exp_proj_var = torch.stack(exp_proj_var, dim=0)
                exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
                exp_proj_var_norm[yaw_axis] = exp_proj_var_norm.max()
                exp_proj_var_norm[pitch_axis] = exp_proj_var_norm.max()
                exp_axis = torch.argmin(exp_proj_var_norm).item()

                anchors_axis_dict = {'yaw_axis':yaw_axis, 'pitch_axis':pitch_axis, 'exp_axis':exp_axis}
                pdb.set_trace()
                
                self.debug_out_path.mkdir(exist_ok=True)
                torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        else:
            anchors_axis_dict = torch.load(self.debug_out_path.joinpath('anchors_axis_dict.pt'))
            yaw_axis = anchors_axis_dict['yaw_axis']
            exp_axis = anchors_axis_dict['exp_axis']
            pitch_axis = anchors_axis_dict['pitch_axis']
            
        # fine-tune
        writer = SummaryWriter(log_dir='runs/{}_{}_yawpitchexp'.format(sample_name.split('_')[0], sample_name.split('_')[1]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            
            tot_loss_anchor_yaw, tot_loss_anchor_exp, tot_loss_anchor_pitch = 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:
        
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # low-dimensional reconstruction
                sample_name = sample.name
                yaw_name = sample_name.split('_')[2]
                pitch_name = sample_name.split('_')[3]
                exp_name = sample_name.split('_')[4]
                anchors_yaw = torch.stack(anchors_yaw_dict[yaw_name], dim=0)
                anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :pca_rank])
                anchors_pitch = torch.stack(anchors_pitch_dict[pitch_name], dim=0)
                anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :pca_rank])    
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :pca_rank])
                
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :pca_rank])

                loss_anchor_yaw = torch.nn.L1Loss()(torch.mean(anchors_yaw_proj, dim=0)[yaw_axis], anchor_proj[0][yaw_axis])
                loss_anchor_pitch = torch.nn.L1Loss()(torch.mean(anchors_pitch_proj, dim=0)[pitch_axis], anchor_proj[0][pitch_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                
                if step < 300:
                    loss_anchor = loss_anchor_yaw
                elif step < 600:
                    loss_anchor = loss_anchor_yaw + loss_anchor_exp
                else:
                    loss_anchor = loss_anchor_yaw + loss_anchor_exp + loss_anchor_pitch
                    
                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_yaw += loss_anchor_yaw.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_pitch += loss_anchor_pitch.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_yaw', tot_loss_anchor_yaw / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_pitch', tot_loss_anchor_pitch / len(dataset), step)
            
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss / len(dataset)):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        writer.close()
        return self.generator
    
    def reconstruct_anchor_yawpitchexpage_celebrities1(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):
        print('reconstruct_anchor_yawpitchexpage_celebrities1')       
        anchors_all_dict, anchors_exp_dict, anchors_yaw_dict, anchors_pitch_dict, anchors_age_dict = {}, {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # eg. barack_obama_VY-16_VP-1_E8_71_align1500.jpg
            yaw_name = sample_name.split('_')[2]
            pitch_name = sample_name.split('_')[3]
            exp_name = sample_name.split('_')[4]
            age_name = sample_name.split('_')[5]

            anchors_all_dict[sample_name] = sample_anchor_param
            if yaw_name in anchors_yaw_dict.keys():
                anchors_yaw_dict[yaw_name].append(sample_anchor_param)
            else:
                anchors_yaw_dict[yaw_name] = [sample_anchor_param]
            
            if pitch_name in anchors_pitch_dict.keys():
                anchors_pitch_dict[pitch_name].append(sample_anchor_param)
            else:
                anchors_pitch_dict[pitch_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]


            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        if resume_epoch == 0:
            with torch.no_grad():
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                yaw_proj_var, pitch_proj_var, exp_proj_var, age_proj_var = [], [], [], []
                rd = pca_rank
                for key in anchors_yaw_dict.keys():
                    anchors_yaw = torch.stack(anchors_yaw_dict[key], dim=0)
                    if anchors_yaw.shape[0] == 1:
                        continue
                    anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                    anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :rd])
                    anchors_yaw_proj_var = torch.var(anchors_yaw_proj, dim=0)
                    yaw_proj_var.append(anchors_yaw_proj_var)
                yaw_proj_var = torch.stack(yaw_proj_var, dim=0)
                yaw_proj_var_norm = torch.sum(yaw_proj_var, dim=0) / S
                yaw_axis = torch.argmin(yaw_proj_var_norm).item()

                for key in anchors_pitch_dict.keys():
                    anchors_pitch = torch.stack(anchors_pitch_dict[key], dim=0)
                    if anchors_pitch.shape[0] == 1:
                        continue
                    anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                    anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :rd])
                    anchors_pitch_proj_var = torch.var(anchors_pitch_proj, dim=0)
                    pitch_proj_var.append(anchors_pitch_proj_var)
                pitch_proj_var = torch.stack(pitch_proj_var, dim=0)
                pitch_proj_var_norm = torch.sum(pitch_proj_var, dim=0) / S
                pitch_proj_var_norm[yaw_axis] = pitch_proj_var_norm.max() + 0.1
                pitch_axis = torch.argmin(pitch_proj_var_norm).item()

                for key in anchors_exp_dict.keys():
                    anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                    if anchors_exp.shape[0] == 1:
                        continue
                    anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                    anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                    anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                    exp_proj_var.append(anchors_exp_proj_var)
                exp_proj_var = torch.stack(exp_proj_var, dim=0)
                exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
                exp_proj_var_norm[yaw_axis] = exp_proj_var_norm.max() + 0.1
                exp_proj_var_norm[pitch_axis] = exp_proj_var_norm.max() + 0.1
                exp_axis = torch.argmin(exp_proj_var_norm).item()

                for key in anchors_age_dict.keys():
                    anchors_age = torch.stack(anchors_age_dict[key], dim=0)
                    if anchors_age.shape[0] == 1:
                        continue
                    anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                    anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                    anchors_age_proj_var = torch.var(anchors_age_proj, dim=0)
                    age_proj_var.append(anchors_age_proj_var)
                age_proj_var = torch.stack(age_proj_var, dim=0)
                age_proj_var_norm = torch.sum(age_proj_var, dim=0) / S
                age_proj_var_norm[yaw_axis] = age_proj_var_norm.max() + 0.1 
                age_proj_var_norm[pitch_axis] = age_proj_var_norm.max() + 0.1
                age_proj_var_norm[exp_axis] = age_proj_var_norm.max() + 0.1
                age_axis = torch.argmin(age_proj_var_norm).item()

                anchors_axis_dict = {'yaw_axis':yaw_axis, 'pitch_axis':pitch_axis, 'exp_axis':exp_axis, 'age_axis':age_axis}
                pdb.set_trace()
                
                self.debug_out_path.mkdir(exist_ok=True)
                torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        else:
            anchors_axis_dict = torch.load(self.debug_out_path.joinpath('anchors_axis_dict.pt'))
            yaw_axis = anchors_axis_dict['yaw_axis']
            exp_axis = anchors_axis_dict['exp_axis']
            age_axis = anchors_axis_dict['age_axis']
            pitch_axis = anchors_axis_dict['pitch_axis']
            
        # fine-tune
        writer = SummaryWriter(log_dir='runs/{}_{}_yawpitchexpage'.format(sample_name.split('_')[0], sample_name.split('_')[1]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            
            tot_loss_anchor_yaw, tot_loss_anchor_pitch, tot_loss_anchor_exp, tot_loss_anchor_age = 0, 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:
        
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # low-dimensional reconstruction
                sample_name = sample.name
                yaw_name = sample_name.split('_')[2]
                pitch_name = sample_name.split('_')[3]
                exp_name = sample_name.split('_')[4]
                age_name = sample_name.split('_')[5]
                anchors_yaw = torch.stack(anchors_yaw_dict[yaw_name], dim=0)
                anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :pca_rank])
                anchors_pitch = torch.stack(anchors_pitch_dict[pitch_name], dim=0)
                anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :pca_rank])    
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :pca_rank])
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :pca_rank])    

                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :pca_rank])

                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                loss_anchor_yaw = torch.nn.L1Loss()(torch.mean(anchors_yaw_proj, dim=0)[yaw_axis], anchor_proj[0][yaw_axis])
                loss_anchor_pitch = torch.nn.L1Loss()(torch.mean(anchors_pitch_proj, dim=0)[pitch_axis], anchor_proj[0][pitch_axis])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[age_axis], anchor_proj[0][age_axis])

                if step < 500:
                    loss_anchor = loss_anchor_exp
                elif step < 1000:
                    loss_anchor = loss_anchor_exp + loss_anchor_age
                elif step < 1500:
                    loss_anchor = loss_anchor_exp + loss_anchor_age + loss_anchor_pitch
                else:
                    loss_anchor = loss_anchor_exp + loss_anchor_age + loss_anchor_pitch + loss_anchor_yaw
                    
                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_age += loss_anchor_age.item()
                tot_loss_anchor_pitch += loss_anchor_pitch.item()
                tot_loss_anchor_yaw += loss_anchor_yaw.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_yaw', tot_loss_anchor_yaw / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_age', tot_loss_anchor_age / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_pitch', tot_loss_anchor_pitch / len(dataset), step)
            
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss / len(dataset)):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        writer.close()
        return self.generator


    def reconstruct_anchor_viewexp_celebrities1(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):
        print('reconstruct_anchor_viewexp_celebrities1')       
        anchors_all_dict, anchors_exp_dict, anchors_view_dict = {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # eg. barack_obama_VY-16_VP-1_E8_71_align1500.jpg
            view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]            
            exp_name = sample_name.split('_')[4]

            anchors_all_dict[sample_name] = sample_anchor_param
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        if resume_epoch == 0:
            with torch.no_grad():
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                view_proj_var, exp_proj_var = [], []
                rd = pca_rank
                for key in anchors_view_dict.keys():
                    anchors_view = torch.stack(anchors_view_dict[key], dim=0)
                    if anchors_view.shape[0] == 1:
                        continue
                    anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                    anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])
                    anchors_view_proj_var = torch.var(anchors_view_proj, dim=0)
                    view_proj_var.append(anchors_view_proj_var)
                view_proj_var = torch.stack(view_proj_var, dim=0)
                view_proj_var_norm = torch.sum(view_proj_var, dim=0) / S
                view_axis = torch.argmin(view_proj_var_norm).item()

                for key in anchors_exp_dict.keys():
                    anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                    if anchors_exp.shape[0] == 1:
                        continue
                    anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                    anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                    anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                    exp_proj_var.append(anchors_exp_proj_var)
                exp_proj_var = torch.stack(exp_proj_var, dim=0)
                exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
                exp_proj_var_norm[view_axis] = exp_proj_var_norm.max()
                exp_axis = torch.argmin(exp_proj_var_norm).item()

                anchors_axis_dict = {'view_axis':view_axis, 'exp_axis':exp_axis}
                pdb.set_trace()
                
                self.debug_out_path.mkdir(exist_ok=True)
                torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        else:
            anchors_axis_dict = torch.load(self.debug_out_path.joinpath('anchors_axis_dict.pt'))
            exp_axis = anchors_axis_dict['exp_axis']
            view_axis = anchors_axis_dict['view_axis']
            
        # fine-tune
        writer = SummaryWriter(log_dir='runs/{}_{}_viewexp'.format(sample_name.split('_')[0], sample_name.split('_')[1]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            
            tot_loss_anchor_exp, tot_loss_anchor_view = 0, 0

            # TODO(1): batched training
            for sample in dataset:
        
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # low-dimensional reconstruction
                sample_name = sample.name
                view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]            
                exp_name = sample_name.split('_')[4]
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :pca_rank])    
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :pca_rank])
                
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :pca_rank])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[view_axis], anchor_proj[0][view_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                
                loss_anchor = loss_anchor_exp + loss_anchor_view
                    
                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_view += loss_anchor_view.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_view', tot_loss_anchor_view / len(dataset), step)
            
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss / len(dataset)):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        writer.close()
        return self.generator


    def reconstruct_anchor_viewexpage_celebrities1(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):
        '''
        compared to previous version,
        automatically find corresponding directions for 
        each attributes.
        '''       
        print('reconstruct_anchor_viewexpage_celebrities1')       
        anchors_all_dict, anchors_exp_dict, anchors_view_dict, anchors_age_dict = {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # eg. emma_watson_VY-7_VP0_ExpS14M14_Age012_Sun04_Mus04_Emma_Watson_72.jpeg
            view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]            
            age_name = sample_name.split('_')[5]
            exp_name = sample_name.split('_')[4]

            anchors_all_dict[sample_name] = sample_anchor_param
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        if resume_epoch == 0:
            with torch.no_grad():
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                view_proj_var, age_proj_var, exp_proj_var = [], [], []
                rd = pca_rank
                for key in anchors_view_dict.keys():
                    anchors_view = torch.stack(anchors_view_dict[key], dim=0)
                    if anchors_view.shape[0] == 1:
                        continue
                    anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                    anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])
                    anchors_view_proj_var = torch.var(anchors_view_proj, dim=0)
                    view_proj_var.append(anchors_view_proj_var)
                view_proj_var = torch.stack(view_proj_var, dim=0)
                view_proj_var_norm = torch.sum(view_proj_var, dim=0) / S
                view_axis = torch.argmin(view_proj_var_norm).item()

                for key in anchors_exp_dict.keys():
                    anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                    if anchors_exp.shape[0] == 1:
                        continue
                    anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                    anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                    anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                    exp_proj_var.append(anchors_exp_proj_var)
                exp_proj_var = torch.stack(exp_proj_var, dim=0)
                exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
                exp_proj_var_norm[view_axis] = exp_proj_var_norm.max() + 0.1
                exp_axis = torch.argmin(exp_proj_var_norm).item()

                for key in anchors_age_dict.keys():
                    anchors_age = torch.stack(anchors_age_dict[key], dim=0)
                    if anchors_age.shape[0] == 1:
                        continue
                    anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                    anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                    anchors_age_proj_var = torch.var(anchors_age_proj, dim=0)
                    age_proj_var.append(anchors_age_proj_var)
                age_proj_var = torch.stack(age_proj_var, dim=0)
                age_proj_var_norm = torch.sum(age_proj_var, dim=0) / S
                age_proj_var_norm[view_axis] = age_proj_var_norm.max() + 0.1
                age_proj_var_norm[exp_axis] = age_proj_var_norm.max() + 0.1
                age_axis = torch.argmin(age_proj_var_norm).item()

                anchors_axis_dict = {'view_axis':view_axis, 'age_axis':age_axis, 'exp_axis':exp_axis}
                pdb.set_trace()
                
                self.debug_out_path.mkdir(exist_ok=True)
                torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        else:
            anchors_axis_dict = torch.load(self.debug_out_path.joinpath('anchors_axis_dict.pt'))
            view_axis = anchors_axis_dict['view_axis']
            exp_axis = anchors_axis_dict['exp_axis']
            age_axis = anchors_axis_dict['age_axis']
            
        # fine-tune
        writer = SummaryWriter(log_dir='runs/{}_{}_viewageexp'.format(sample_name.split('_')[0], sample_name.split('_')[1]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            
            tot_loss_anchor_view, tot_loss_anchor_exp, tot_loss_anchor_age = 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:
        
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # low-dimensional reconstruction
                sample_name = sample.name
                view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]            
                age_name = sample_name.split('_')[5]
                exp_name = sample_name.split('_')[4]
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :pca_rank])
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :pca_rank])    
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :pca_rank])
                
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :pca_rank])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[view_axis], anchor_proj[0][view_axis])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[age_axis], anchor_proj[0][age_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                
                if step < 300:
                    loss_anchor = loss_anchor_view
                elif step < 600:
                    loss_anchor = loss_anchor_view + loss_anchor_exp
                else:
                    loss_anchor = loss_anchor_view + loss_anchor_exp + loss_anchor_age
                    
                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_view += loss_anchor_view.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_age += loss_anchor_age.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_view', tot_loss_anchor_view / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_age', tot_loss_anchor_age / len(dataset), step)
            
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss / len(dataset)):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        writer.close()
        return self.generator



    def reconstruct_anchor_auto_dir_celebrities1_ageeye(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False):
        '''
        compared to previous version,
        automatically find corresponding directions for 
        each attributes.
        attributes: yaw, pitch, exp, age, eye
        '''       
        print('reconstruct_anchor_auto_dir_celebrities1_ageeye')       
        anchors_all_dict, anchors_exp_dict, anchors_yaw_dict, anchors_pitch_dict, anchors_age_dict, anchors_eye_dict = {}, {}, {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # eg. barack_obama_VY-16_VP-1_E8_71_align1500.jpg
            yaw_name = sample_name.split('_')[2]
            pitch_name = sample_name.split('_')[3]
            exp_name = sample_name.split('_')[4]
            age_name = sample_name.split('_')[6]
            eye_name = sample_name.split('_')[7]

            anchors_all_dict[sample_name] = sample_anchor_param
            if yaw_name in anchors_yaw_dict.keys():
                anchors_yaw_dict[yaw_name].append(sample_anchor_param)
            else:
                anchors_yaw_dict[yaw_name] = [sample_anchor_param]
            
            if pitch_name in anchors_pitch_dict.keys():
                anchors_pitch_dict[pitch_name].append(sample_anchor_param)
            else:
                anchors_pitch_dict[pitch_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]
            
            if eye_name in anchors_eye_dict.keys():
                anchors_eye_dict[eye_name].append(sample_anchor_param)
            else:
                anchors_eye_dict[eye_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        with torch.no_grad():
            anchors_params_all = torch.stack(anchors_param_list, dim=0)
            X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
            U, S, V = torch.pca_lowrank(X, q=pca_rank)
            yaw_proj_var, pitch_proj_var, exp_proj_var, age_proj_var, eye_proj_var = [], [], [], [], []
            rd = pca_rank
            for key in anchors_yaw_dict.keys():
                anchors_yaw = torch.stack(anchors_yaw_dict[key], dim=0)
                if anchors_yaw.shape[0] == 1:
                    continue
                anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :rd])
                anchors_yaw_proj_var = torch.var(anchors_yaw_proj, dim=0)
                yaw_proj_var.append(anchors_yaw_proj_var)
            yaw_proj_var = torch.stack(yaw_proj_var, dim=0)
            yaw_proj_var_norm = torch.sum(yaw_proj_var, dim=0) / S
            yaw_axis = torch.argmin(yaw_proj_var_norm).item()

            for key in anchors_pitch_dict.keys():
                anchors_pitch = torch.stack(anchors_pitch_dict[key], dim=0)
                if anchors_pitch.shape[0] == 1:
                    continue
                anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :rd])
                anchors_pitch_proj_var = torch.var(anchors_pitch_proj, dim=0)
                pitch_proj_var.append(anchors_pitch_proj_var)
            pitch_proj_var = torch.stack(pitch_proj_var, dim=0)
            pitch_proj_var_norm = torch.sum(pitch_proj_var, dim=0) / S
            pitch_proj_var_norm[yaw_axis] = pitch_proj_var_norm.max()
            pitch_axis = torch.argmin(pitch_proj_var_norm).item()

            for key in anchors_exp_dict.keys():
                anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                if anchors_exp.shape[0] == 1:
                    continue
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                exp_proj_var.append(anchors_exp_proj_var)
            exp_proj_var = torch.stack(exp_proj_var, dim=0)
            exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
            exp_proj_var_norm[yaw_axis] = exp_proj_var_norm.max()
            exp_proj_var_norm[pitch_axis] = exp_proj_var_norm.max()
            exp_axis = torch.argmin(exp_proj_var_norm).item()

            for key in anchors_age_dict.keys():
                anchors_age = torch.stack(anchors_age_dict[key], dim=0)
                if anchors_age.shape[0] == 1:
                    continue
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_age_proj_var = torch.var(anchors_age_proj, dim=0)
                age_proj_var.append(anchors_age_proj_var)
            age_proj_var = torch.stack(age_proj_var, dim=0)
            age_proj_var_norm = torch.sum(age_proj_var, dim=0) / S
            age_proj_var_norm[yaw_axis] = age_proj_var_norm.max()
            age_proj_var_norm[pitch_axis] = age_proj_var_norm.max()
            age_proj_var_norm[exp_axis] = age_proj_var_norm.max()
            age_axis = torch.argmin(age_proj_var_norm).item()

            for key in anchors_eye_dict.keys():
                anchors_eye = torch.stack(anchors_eye_dict[key], dim=0)
                if anchors_eye.shape[0] == 1:
                    continue
                anchors_eye = anchors_eye.reshape(anchors_eye.shape[0], -1)
                anchors_eye_proj = torch.matmul(anchors_eye, V[:, :rd])
                anchors_eye_proj_var = torch.var(anchors_eye_proj, dim=0)
                eye_proj_var.append(anchors_eye_proj_var)
            eye_proj_var = torch.stack(eye_proj_var, dim=0)
            eye_proj_var_norm = torch.sum(eye_proj_var, dim=0) / S
            eye_proj_var_norm[yaw_axis] = eye_proj_var_norm.max()
            eye_proj_var_norm[pitch_axis] = eye_proj_var_norm.max()
            eye_proj_var_norm[exp_axis] = eye_proj_var_norm.max()
            eye_proj_var_norm[age_axis] = eye_proj_var_norm.max()
            eye_axis = torch.argmin(eye_proj_var_norm).item()

            anchors_axis_dict = {'yaw_axis':yaw_axis, 'pitch_axis':pitch_axis, 'exp_axis':exp_axis, 'age_axis':age_axis, 'eye_axis':eye_axis}
            pdb.set_trace()
            
            self.debug_out_path.mkdir(exist_ok=True)
            torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        
        # fine-tune
        writer = SummaryWriter()
        for step in tqdm(range(self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)

            tot_loss_anchor_yaw, tot_loss_anchor_eye, tot_loss_anchor_exp, tot_loss_anchor_pitch, tot_loss_anchor_age = 0, 0, 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:
        
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # low-dimensional reconstruction
                sample_name = sample.name
                yaw_name = sample_name.split('_')[2]
                pitch_name = sample_name.split('_')[3]
                exp_name = sample_name.split('_')[4]
                anchors_yaw = torch.stack(anchors_yaw_dict[yaw_name], dim=0)
                anchors_yaw = anchors_yaw.reshape(anchors_yaw.shape[0], -1)
                anchors_yaw_proj = torch.matmul(anchors_yaw, V[:, :rd])
                anchors_pitch = torch.stack(anchors_pitch_dict[pitch_name], dim=0)
                anchors_pitch = anchors_pitch.reshape(anchors_pitch.shape[0], -1)
                anchors_pitch_proj = torch.matmul(anchors_pitch, V[:, :rd])    
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_eye = torch.stack(anchors_eye_dict[eye_name], dim=0)
                anchors_eye = anchors_eye.reshape(anchors_eye.shape[0], -1)
                anchors_eye_proj = torch.matmul(anchors_eye, V[:, :rd])

                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_yaw = torch.nn.L1Loss()(torch.mean(anchors_yaw_proj, dim=0)[yaw_axis], anchor_proj[0][yaw_axis])
                loss_anchor_eye = torch.nn.L1Loss()(torch.mean(anchors_eye_proj, dim=0)[eye_axis], anchor_proj[0][eye_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                loss_anchor_pitch = torch.nn.L1Loss()(torch.mean(anchors_pitch_proj, dim=0)[pitch_axis], anchor_proj[0][pitch_axis])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[age_axis], anchor_proj[0][age_axis])
                
                if step < 300:
                    loss_anchor = loss_anchor_yaw
                elif step < 600:
                    loss_anchor = loss_anchor_yaw + loss_anchor_eye
                elif step < 900:
                    loss_anchor = loss_anchor_yaw + loss_anchor_eye + loss_anchor_exp
                elif step < 1200:
                    loss_anchor = loss_anchor_yaw + loss_anchor_eye + loss_anchor_exp + loss_anchor_pitch
                else:
                    loss_anchor = loss_anchor_yaw + loss_anchor_eye + loss_anchor_exp + loss_anchor_pitch + loss_anchor_age
                    
                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_yaw += loss_anchor_yaw.item()
                tot_loss_anchor_eye += loss_anchor_eye.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_pitch += loss_anchor_pitch.item()
                tot_loss_anchor_age += loss_anchor_age.item()

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_yaw', tot_loss_anchor_yaw / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_eye', tot_loss_anchor_eye / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_pitch', tot_loss_anchor_pitch / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_age', tot_loss_anchor_age / len(dataset), step)
            
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss / len(dataset)):<5.2f}')
            
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        writer.close()

        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        return self.generator
    
    
    def reconstruct_anchor_age_auto_dir(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, resume_epoch=0):
        '''
        compared to previous version,
        automatically find corresponding directions for 
        each attributes.
        '''
        print('reconstruct_anchor_age_auto_dir')       
        anchors_all_dict, anchors_exp_dict, anchors_age_dict, anchors_view_dict = {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            age_name = sample_name.split('_')[0]
            exp_name = sample_name.split('_')[1]
            view_name = sample_name.split('_')[2]

            anchors_all_dict[sample_name] = sample_anchor_param
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        if resume_epoch == 0:
            with torch.no_grad():
                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                age_proj_var, view_proj_var, exp_proj_var = [], [], []
                rd = pca_rank
                for key in anchors_view_dict.keys():
                    anchors_view = torch.stack(anchors_view_dict[key], dim=0)
                    if anchors_view.shape[0] == 1:
                        continue
                    anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                    anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])
                    anchors_view_proj_var = torch.var(anchors_view_proj, dim=0)
                    view_proj_var.append(anchors_view_proj_var)
                view_proj_var = torch.stack(view_proj_var, dim=0)
                view_proj_var_norm = torch.sum(view_proj_var, dim=0) / S
                view_axis = torch.argmin(view_proj_var_norm).item()

                for key in anchors_exp_dict.keys():
                    anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                    if anchors_exp.shape[0] == 1:
                        continue
                    anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                    anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                    anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                    exp_proj_var.append(anchors_exp_proj_var)
                exp_proj_var = torch.stack(exp_proj_var, dim=0)
                exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
                exp_proj_var_norm[view_axis] = exp_proj_var_norm.max()
                exp_axis = torch.argmin(exp_proj_var_norm).item()

                for key in anchors_age_dict.keys():
                    anchors_age = torch.stack(anchors_age_dict[key], dim=0)
                    if anchors_age.shape[0] == 1:
                        continue
                    anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                    anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                    anchors_age_proj_var = torch.var(anchors_age_proj, dim=0)
                    age_proj_var.append(anchors_age_proj_var)
                age_proj_var = torch.stack(age_proj_var, dim=0)
                age_proj_var_norm = torch.sum(age_proj_var, dim=0) / S
                age_proj_var_norm[view_axis] = age_proj_var_norm.max()
                age_proj_var_norm[exp_axis] = age_proj_var_norm.max()
                age_axis = torch.argmin(age_proj_var_norm).item()
                
                anchors_axis_dict = {'view_axis':view_axis, 'exp_axis':exp_axis, 'age_axis':age_axis}
                self.debug_out_path.mkdir(exist_ok=True)
                torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        else:
            anchors_axis_dict = torch.load(self.debug_out_path.joinpath('anchors_axis_dict.pt'))
            view_axis = anchors_axis_dict['view_axis']
            exp_axis = anchors_axis_dict['exp_axis']
            age_axis = anchors_axis_dict['age_axis']

        # fine-tune
        writer = SummaryWriter(log_dir='runs/{}_{}_age_smc_viewageexp'.format(sample_name.split('_')[3], sample_name.split('_')[4]))
        for step in tqdm(range(resume_epoch, self.num_steps)):
            tot_loss, tot_loss_recon = 0, 0
            to_visualize = self.need_visualize(step)
            
            tot_loss_anchor_view, tot_loss_anchor_exp, tot_loss_anchor_age = 0, 0, 0

            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                sample_name = sample.name
                age_name = sample_name.split('_')[0]
                exp_name = sample_name.split('_')[1]
                view_name = sample_name.split('_')[2]
                pdb.set_trace()
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[view_axis], anchor_proj[0][view_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[age_axis], anchor_proj[0][age_axis])
                loss_anchor = loss_anchor_age + loss_anchor_exp + loss_anchor_view
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                tot_loss_recon += loss_recon.item()
                tot_loss_anchor_view += loss_anchor_view.item()
                tot_loss_anchor_exp += loss_anchor_exp.item() 
                tot_loss_anchor_age += loss_anchor_age.item()

                # print('pca {} timing:{}'.format(pca_rank, pca_ed_time - pca_st_time))
                # print('anchor loss {} timing:{}'.format(pca_rank, anchor_loss_ed_time - pca_ed_time))
                # print('recon loss {} timing:{}'.format(pca_rank, recon_loss_ed_time - anchor_loss_ed_time))

                # if cnt == 4:
                #     pdb.set_trace()
                # cnt += 1

            writer.add_scalar('Loss/tot_loss', tot_loss / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_recon', tot_loss_recon / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_view', tot_loss_anchor_view / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_exp', tot_loss_anchor_exp / len(dataset), step)
            writer.add_scalar('Loss/tot_loss_anchor_age', tot_loss_anchor_age / len(dataset), step)
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')

            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        writer.close()
        return self.generator

    def reconstruct_anchor_age_sun_mus_auto_dir(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False):
        '''
        compared to previous version,
        automatically find corresponding directions for 
        each attributes.
        '''       
        anchors_all_dict, anchors_exp_dict, anchors_age_dict, anchors_view_dict, anchors_sun_dict, anchors_mus_dict = {}, {}, {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            age_name = sample_name.split('_')[0]
            exp_name = sample_name.split('_')[1]
            view_name = sample_name.split('_')[2]
            sun_name = sample_name.split('_')[3]
            mus_name = sample_name.split('_')[4]

            anchors_all_dict[sample_name] = sample_anchor_param
            if age_name in anchors_age_dict.keys():
                anchors_age_dict[age_name].append(sample_anchor_param)
            else:
                anchors_age_dict[age_name] = [sample_anchor_param]
            
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]
            
            if sun_name in anchors_sun_dict.keys():
                anchors_sun_dict[sun_name].append(sample_anchor_param)
            else:
                anchors_sun_dict[sun_name] = [sample_anchor_param]
            
            if mus_name in anchors_mus_dict.keys():
                anchors_mus_dict[mus_name].append(sample_anchor_param)
            else:
                anchors_mus_dict[mus_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)
        
        # find corresponding directions for each attribute
        with torch.no_grad():
            anchors_params_all = torch.stack(anchors_param_list, dim=0)
            X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
            U, S, V = torch.pca_lowrank(X, q=pca_rank)
            age_proj_var, view_proj_var, exp_proj_var, sun_proj_var, mus_proj_var = [], [], [], [], []
            rd = pca_rank
            for key in anchors_view_dict.keys():
                anchors_view = torch.stack(anchors_view_dict[key], dim=0)
                if anchors_view.shape[0] == 1:
                    continue
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])
                anchors_view_proj_var = torch.var(anchors_view_proj, dim=0)
                view_proj_var.append(anchors_view_proj_var)
            view_proj_var = torch.stack(view_proj_var, dim=0)
            view_proj_var_norm = torch.sum(view_proj_var, dim=0) / S
            view_axis = torch.argmin(view_proj_var_norm).item()

            for key in anchors_exp_dict.keys():
                anchors_exp = torch.stack(anchors_exp_dict[key], dim=0)
                if anchors_exp.shape[0] == 1:
                    continue
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_exp_proj_var = torch.var(anchors_exp_proj, dim=0)
                exp_proj_var.append(anchors_exp_proj_var)
            exp_proj_var = torch.stack(exp_proj_var, dim=0)
            exp_proj_var_norm = torch.sum(exp_proj_var, dim=0) / S
            exp_proj_var_norm[view_axis] = exp_proj_var_norm.max()
            exp_axis = torch.argmin(exp_proj_var_norm).item()

            for key in anchors_age_dict.keys():
                anchors_age = torch.stack(anchors_age_dict[key], dim=0)
                if anchors_age.shape[0] == 1:
                    continue
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_age_proj_var = torch.var(anchors_age_proj, dim=0)
                age_proj_var.append(anchors_age_proj_var)
            age_proj_var = torch.stack(age_proj_var, dim=0)
            age_proj_var_norm = torch.sum(age_proj_var, dim=0) / S
            age_proj_var_norm[view_axis] = age_proj_var_norm.max()
            age_proj_var_norm[exp_axis] = age_proj_var_norm.max()
            age_axis = torch.argmin(age_proj_var_norm).item()

            for key in anchors_sun_dict.keys():
                anchors_sun = torch.stack(anchors_sun_dict[key], dim=0)
                if anchors_sun.shape[0] == 1:
                    continue
                anchors_sun = anchors_sun.reshape(anchors_sun.shape[0], -1)
                anchors_sun_proj = torch.matmul(anchors_sun, V[:, :rd])
                anchors_sun_proj_var = torch.var(anchors_sun_proj, dim=0)
                sun_proj_var.append(anchors_sun_proj_var)
            sun_proj_var = torch.stack(sun_proj_var, dim=0)
            sun_proj_var_norm = torch.sum(sun_proj_var, dim=0) / S
            sun_proj_var_norm[view_axis] = sun_proj_var_norm.max()
            sun_proj_var_norm[exp_axis] = sun_proj_var_norm.max()
            sun_proj_var_norm[age_axis] = sun_proj_var_norm.max()
            sun_axis = torch.argmin(sun_proj_var_norm).item()

            for key in anchors_mus_dict.keys():
                anchors_mus = torch.stack(anchors_mus_dict[key], dim=0)
                if anchors_mus.shape[0] == 1:
                    continue
                anchors_mus = anchors_mus.reshape(anchors_mus.shape[0], -1)
                anchors_mus_proj = torch.matmul(anchors_mus, V[:, :rd])
                anchors_mus_proj_var = torch.var(anchors_mus_proj, dim=0)
                mus_proj_var.append(anchors_mus_proj_var)
            mus_proj_var = torch.stack(mus_proj_var, dim=0)
            mus_proj_var_norm = torch.sum(mus_proj_var, dim=0) / S
            mus_proj_var_norm[view_axis] = mus_proj_var_norm.max()
            mus_proj_var_norm[exp_axis] = mus_proj_var_norm.max()
            mus_proj_var_norm[age_axis] = mus_proj_var_norm.max()
            mus_proj_var_norm[sun_axis] = mus_proj_var_norm.max()
            mus_axis = torch.argmin(mus_proj_var_norm).item()

            anchors_axis_dict = {'view_axis':view_axis, 'exp_axis':exp_axis, 'age_axis':age_axis, 'sun_axis':sun_axis, 'mus_axis':mus_axis}
            self.debug_out_path.mkdir(exist_ok=True)
            torch.save(anchors_axis_dict, self.debug_out_path.joinpath('anchors_axis_dict.pt'))
        
        # fine-tune
        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                sample_name = sample.name
                age_name = sample_name.split('_')[0]
                exp_name = sample_name.split('_')[1]
                view_name = sample_name.split('_')[2]
                if is_lightstage:
                    rd = 3
                    light_name = sample_name.split('_')[-1]
                    anchors_light = torch.stack(anchors_light_dict[light_name], dim=0)
                    anchors_light = anchors_light.reshape(anchors_light.shape[0], -1)
                    anchors_light_proj = torch.matmul(anchors_light, V[:, :rd])
                anchors_age = torch.stack(anchors_age_dict[age_name], dim=0)
                anchors_age = anchors_age.reshape(anchors_age.shape[0], -1)
                anchors_age_proj = torch.matmul(anchors_age, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])    
                anchors_sun = torch.stack(anchors_sun_dict[sun_name], dim=0)
                anchors_sun = anchors_sun.reshape(anchors_sun.shape[0], -1)
                anchors_sun_proj = torch.matmul(anchors_sun, V[:, :rd])    
                anchors_mus = torch.stack(anchors_mus_dict[mus_name], dim=0)
                anchors_mus = anchors_mus.reshape(anchors_mus.shape[0], -1)
                anchors_mus_proj = torch.matmul(anchors_mus, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[view_axis], anchor_proj[0][view_axis])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[exp_axis], anchor_proj[0][exp_axis])
                loss_anchor_age = torch.nn.L1Loss()(torch.mean(anchors_age_proj, dim=0)[age_axis], anchor_proj[0][age_axis])
                loss_anchor_sun = torch.nn.L1Loss()(torch.mean(anchors_sun_proj, dim=0)[sun_axis], anchor_proj[0][sun_axis])
                loss_anchor_mus = torch.nn.L1Loss()(torch.mean(anchors_mus_proj, dim=0)[mus_axis], anchor_proj[0][mus_axis])
                loss_anchor = loss_anchor_age + loss_anchor_exp + loss_anchor_view + loss_anchor_sun + loss_anchor_mus
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                
                # print('pca {} timing:{}'.format(pca_rank, pca_ed_time - pca_st_time))
                # print('anchor loss {} timing:{}'.format(pca_rank, anchor_loss_ed_time - pca_ed_time))
                # print('recon loss {} timing:{}'.format(pca_rank, recon_loss_ed_time - anchor_loss_ed_time))

                # if cnt == 4:
                #     pdb.set_trace()
                # cnt += 1
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        return self.generator

    def reconstruct_anchor_fb(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False, is_wspace=False, is_yaw=True):        
        view_dict_path = '/nfs/STG/CodecAvatar/lelechen/libingzeng/mystyle_lb/data/fb_multifaces/samples/view_dict.pickle'
        with open(view_dict_path, 'rb') as handle: 
            view_dict = pickle.load(handle)

        anchors_all_dict, anchors_exp_dict, anchors_view_dict = {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            if not is_wspace:
                sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device) # w+
            else:
                sample_anchor_param = torch.nn.Parameter(sample.w_code[:, :1, :]).to(self.device) # w
            sample_name = sample.name
            # exp_name = sample_name.split('_')[4] # new celebrities
            # view_name = sample_name.split('_')[2] + '_' + sample_name.split('_')[3]
            exp_name = sample_name.split('_')[1] # fb_multifaces
            view_key = sample_name.split('_')[-2]
            if view_key not in view_dict.keys():
                continue
            if is_yaw:
                view_name = view_dict[view_key].split('_')[0]
            else:
                view_name = view_dict[view_key].split('_')[1]
            
            anchors_all_dict[sample_name] = sample_anchor_param
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if view_name in anchors_view_dict.keys():
                anchors_view_dict[view_name].append(sample_anchor_param)
            else:
                anchors_view_dict[view_name] = [sample_anchor_param]

            if is_lightstage:
                light_name = sample_name.split('_')[-1].split('.')[0]
                if light_name in anchors_light_dict.keys():
                    anchors_light_dict[light_name].append(sample_anchor_param)
                else:
                    anchors_light_dict[light_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)

        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                rd = 2 # number of reduced dimensions to use
                sample_name = sample.name
                exp_name = sample_name.split('_')[1] # fb_multifaces
                view_key = sample_name.split('_')[-2]
                if view_key not in view_dict.keys():
                    continue
                if is_yaw:
                    view_name = view_dict[view_key].split('_')[0]
                else:
                    view_name = view_dict[view_key].split('_')[1]
                if is_lightstage:
                    rd = 3
                    light_name = sample_name.split('_')[-1]
                    anchors_light = torch.stack(anchors_light_dict[light_name], dim=0)
                    anchors_light = anchors_light.reshape(anchors_light.shape[0], -1)
                    anchors_light_proj = torch.matmul(anchors_light, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_view = torch.stack(anchors_view_dict[view_name], dim=0)
                anchors_view = anchors_view.reshape(anchors_view.shape[0], -1)
                anchors_view_proj = torch.matmul(anchors_view, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_view = torch.nn.L1Loss()(torch.mean(anchors_view_proj, dim=0)[0], anchor_proj[0][0])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[1], anchor_proj[0][1])
                if is_lightstage:
                    loss_anchor_light = torch.nn.L1Loss()(torch.mean(anchors_light_proj, dim=0)[2], anchor_proj[0][2])
                    loss_anchor = loss_anchor_exp + loss_anchor_view + loss_anchor_light
                else:
                    loss_anchor = loss_anchor_exp + loss_anchor_view
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                if not is_wspace:
                    synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                else:
                    synth = self.generator(anchor.expand(sample.w_code.shape), noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                
                # print('pca {} timing:{}'.format(pca_rank, pca_ed_time - pca_st_time))
                # print('anchor loss {} timing:{}'.format(pca_rank, anchor_loss_ed_time - pca_ed_time))
                # print('recon loss {} timing:{}'.format(pca_rank, recon_loss_ed_time - anchor_loss_ed_time))

                # if cnt == 4:
                #     pdb.set_trace()
                # cnt += 1
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))
        
        torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dict_{}.pt'.format(str(step).zfill(4))))
        torch.save(self.generator, self.debug_out_path.joinpath('mystyle_model_step{}.pt'.format(str(step).zfill(4))))

        return self.generator

    def reconstruct_anchor_vyvp(self, dataset, pca_rank=6, pca_extra_dim=0, is_lightstage=False, is_pca_remaining_opt=False):        
        '''
        view is separated in yaw and pitch, 
        and enforce the first two components as views, and the third as expression
        '''
        print('reconstruct_anchor_vyvp')
        anchors_all_dict, anchors_exp_dict, anchors_viewy_dict, anchors_viewp_dict = {}, {}, {}, {}
        if is_lightstage:
            anchors_light_dict = {}
        anchors_param_list = []
        for sample in dataset:
            sample_anchor_param = torch.nn.Parameter(sample.w_code).to(self.device)
            sample_name = sample.name
            exp_name = sample_name.split('_')[-2]
            viewy_name = sample_name.split('_')[1]
            viewp_name = sample_name.split('_')[2]
            anchors_all_dict[sample_name] = sample_anchor_param
            if exp_name in anchors_exp_dict.keys():
                anchors_exp_dict[exp_name].append(sample_anchor_param)
            else:
                anchors_exp_dict[exp_name] = [sample_anchor_param]
            
            if viewy_name in anchors_viewy_dict.keys():
                anchors_viewy_dict[viewy_name].append(sample_anchor_param)
            else:
                anchors_viewy_dict[viewy_name] = [sample_anchor_param]
            
            if viewp_name in anchors_viewp_dict.keys():
                anchors_viewp_dict[viewp_name].append(sample_anchor_param)
            else:
                anchors_viewp_dict[viewp_name] = [sample_anchor_param]

            if is_lightstage:
                light_name = sample_name.split('_')[-1].split('.')[0]
                if light_name in anchors_light_dict.keys():
                    anchors_light_dict[light_name].append(sample_anchor_param)
                else:
                    anchors_light_dict[light_name] = [sample_anchor_param]

            anchors_param_list.append(sample_anchor_param)
            
        self.optimizer = torch.optim.Adam(anchors_param_list + list(self.generator.parameters()), lr=hyperparams.tune_lr)

        for step in tqdm(range(self.num_steps)):
            tot_loss = 0
            to_visualize = self.need_visualize(step)

            # TODO(1): batched training
            for sample in dataset:

                anchors_params_all = torch.stack(anchors_param_list, dim=0)
                X = anchors_params_all.reshape(anchors_params_all.shape[0], -1)
                # pca_st_time = time.time()
                U, S, V = torch.pca_lowrank(X, q=pca_rank)
                # pca_ed_time = time.time()
                # low-dimensional reconstruction
                rd = 3 # number of reduced dimensions to use
                sample_name = sample.name
                exp_name = sample_name.split('_')[-2]
                viewy_name = sample_name.split('_')[1]
                viewp_name = sample_name.split('_')[2]
                if is_lightstage:
                    rd = 4
                    light_name = sample_name.split('_')[-1]
                    anchors_light = torch.stack(anchors_light_dict[light_name], dim=0)
                    anchors_light = anchors_light.reshape(anchors_light.shape[0], -1)
                    anchors_light_proj = torch.matmul(anchors_light, V[:, :rd])
                anchors_exp = torch.stack(anchors_exp_dict[exp_name], dim=0)
                anchors_exp = anchors_exp.reshape(anchors_exp.shape[0], -1)
                anchors_exp_proj = torch.matmul(anchors_exp, V[:, :rd])
                anchors_viewy = torch.stack(anchors_viewy_dict[viewy_name], dim=0)
                anchors_viewy = anchors_viewy.reshape(anchors_viewy.shape[0], -1)
                anchors_viewy_proj = torch.matmul(anchors_viewy, V[:, :rd])    
                anchors_viewp = torch.stack(anchors_viewp_dict[viewp_name], dim=0)
                anchors_viewp = anchors_viewp.reshape(anchors_viewp.shape[0], -1)
                anchors_viewp_proj = torch.matmul(anchors_viewp, V[:, :rd])    
                anchor = anchors_all_dict[sample_name]
                anchor_proj = torch.matmul(anchor.reshape(anchor.shape[0], -1), V[:, :rd])

                loss_anchor_viewy = torch.nn.L1Loss()(torch.mean(anchors_viewy_proj, dim=0)[0], anchor_proj[0][0])
                loss_anchor_viewp = torch.nn.L1Loss()(torch.mean(anchors_viewp_proj, dim=0)[1], anchor_proj[0][1])
                loss_anchor_exp = torch.nn.L1Loss()(torch.mean(anchors_exp_proj, dim=0)[2], anchor_proj[0][2])
                if is_lightstage:
                    loss_anchor_light = torch.nn.L1Loss()(torch.mean(anchors_light_proj, dim=0)[2], anchor_proj[0][2])
                    loss_anchor = loss_anchor_exp + loss_anchor_viewy + loss_anchor_viewp + loss_anchor_light
                else:
                    loss_anchor = loss_anchor_exp + loss_anchor_viewy + loss_anchor_viewp
                
                if is_pca_remaining_opt:
                    loss_anchor_others = S[(rd + pca_extra_dim):].sum() / S.sum() # one (pca_extra_dim) extra dimension to encode other information of dataset
                    loss_anchor += loss_anchor_others
                # anchor_loss_ed_time = time.time()

                target = sample.img.cuda()
                synth = self.generator(anchor, noise_mode='const', force_fp32=True)
                loss_recon = self.reconstruction_loss(synth, target)
                # recon_loss_ed_time = time.time()

                loss = loss_recon + loss_anchor * 0.01
                if to_visualize:
                    io_utils.save_images(synth, self.debug_out_path.joinpath(sample.name))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()
                
                # print('pca {} timing:{}'.format(pca_rank, pca_ed_time - pca_st_time))
                # print('anchor loss {} timing:{}'.format(pca_rank, anchor_loss_ed_time - pca_ed_time))
                # print('recon loss {} timing:{}'.format(pca_rank, recon_loss_ed_time - anchor_loss_ed_time))

                # if cnt == 4:
                #     pdb.set_trace()
                # cnt += 1
            print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(tot_loss):<5.2f}')
            if step % 100 == 0:
                torch.save(anchors_all_dict, self.debug_out_path.joinpath('anchors_all_dictvyvp_{}.pt'.format(str(step).zfill(4))))
                torch.save(self.generator, self.debug_out_path.joinpath('mystyle_modelvyvp_step{}.pt'.format(str(step).zfill(4))))

        return self.generator
