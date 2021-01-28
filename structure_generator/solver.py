import time
import deepspeed as ds
from pathlib import Path
from PIL import Image

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm

from models import *
from utils import (
        Colorize,
        return_colormap,
        load_dataset,
        normalize_data_dp,
        plot,
        )


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def tprint_rank_0(tqdm, message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            tqdm.set_description(message)
    else:
        tqdm.set_description(message)


class Solver(object):
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.output_dir = Path(opt.output_dir) / self.name
        self.preddump_dir = self.output_dir / 'preddump'
        self.preddump_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = self.output_dir / 'sample'
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'tensorboard'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / 'ckpt'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_iter = 0
        self.init_loss_functions()
        self.init_colorize()
        self.init_models_optimizers_data()
        self.load_states()

        if not opt.inference:
            self.writer = SummaryWriter(self.log_dir, purge_step=self.global_iter)

    def init_models_optimizers_data(self):
        opt = self.opt
        device = opt.device
        self.encoder, self.decoder = get_autoencoder(opt)
        self.frame_predictor = DeterministicConvLSTM(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size, opt.M)
        self.posterior = GaussianConvLSTM(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size, opt.M)
        self.prior = GaussianConvLSTM(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size, opt.M)
        if not opt.deepspeed:
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            self.frame_predictor = self.frame_predictor.to(device)
            self.posterior = self.posterior.to(device)
            self.prior = self.prior.to(device)

        self.frame_predictor_optimizer = optim.Adam(self.frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.posterior_optimizer = optim.Adam(self.posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.prior_optimizer = optim.Adam(self.prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.frame_predictor.apply(init_weights)
        self.posterior.apply(init_weights)
        self.prior.apply(init_weights)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        encoder_params = filter(lambda p: p.requires_grad, self.encoder.parameters())
        decoder_params = filter(lambda p: p.requires_grad, self.decoder.parameters())
        frame_predictor_params = filter(lambda p: p.requires_grad, self.frame_predictor.parameters())
        posterior_params = filter(lambda p: p.requires_grad, self.posterior.parameters())
        prior_params = filter(lambda p: p.requires_grad, self.prior.parameters())

        if opt.load_dp_ckpt:
            self.load_dp_ckpt()
        if opt.load_ds_ckpt:
            self.load_ds_ckpt()

        train_data, test_data = load_dataset(opt)
        if opt.inference:
            # use pytorch loaders for both train/test loader in inference mode
            train_loader = DataLoader(train_data,
                                      num_workers=opt.data_threads,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
            test_loader = DataLoader(test_data,
                                     num_workers=opt.data_threads,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     pin_memory=True)
        elif not opt.inference and not opt.deepspeed:
            # use pytorch loaders for both train/test loader when not using deepspeed
            train_loader = DataLoader(train_data,
                                      num_workers=opt.data_threads,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
            test_loader = DataLoader(test_data,
                                     num_workers=opt.data_threads,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     pin_memory=True)
        elif not opt.inference and opt.deepspeed:
            # use deepspeed train loader when training with deepspeed.
            # use pytorch test loader when testing
            test_loader = DataLoader(test_data,
                                     num_workers=opt.data_threads,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     pin_memory=True)

        if opt.deepspeed:
            if not opt.inference:
                self.encoder, self.encoder_optimizer, train_loader, _ = ds.initialize(opt, model=self.encoder, model_parameters=encoder_params, dist_init_required=True, training_data=train_data)
            else:
                self.encoder, self.encoder_optimizer, _, _ = ds.initialize(opt, model=self.encoder, model_parameters=encoder_params, dist_init_required=True)
            self.decoder, self.decoder_optimizer, _, _ = ds.initialize(opt, model=self.decoder, model_parameters=decoder_params, dist_init_required=False)
            self.frame_predictor, self.frame_predictor_optimizer, _, _ = ds.initialize(opt, model=self.frame_predictor, model_parameters=frame_predictor_params, dist_init_required=False)
            self.posterior, self.posterior_optimizer, _, _ = ds.initialize(opt, model=self.posterior, model_parameters=posterior_params, dist_init_required=False)
            self.prior, self.prior_optimizer, _, _ = ds.initialize(opt, model=self.prior, model_parameters=prior_params, dist_init_required=False)

            def normalize_data_ds(opt, sequence):
                data, data_path = sequence
                data.transpose_(0, 1)
                return data.to(self.encoder.local_rank), data_path
            def get_batch(loader):
                while True:
                    for sequence in loader:
                        batch = normalize_data_ds(opt, sequence)
                        yield batch
            def get_dump_batch(loader):
                for sequence in loader:
                    batch = normalize_data_ds(opt, sequence)
                    yield batch
        else:
            self.encoder = DataParallel(self.encoder)
            self.decoder = DataParallel(self.decoder)
            self.frame_predictor = DataParallel(self.frame_predictor)
            self.posterior = DataParallel(self.posterior)
            self.prior = DataParallel(self.prior)

            if opt.device == 'cuda':
                dtype = torch.cuda.FloatTensor
            else:
                dtype = torch.FloatTensor
            def get_batch(loader):
                while True:
                    for sequence in loader:
                        batch = normalize_data_dp(opt, dtype, sequence)
                        yield batch
            def get_dump_batch(loader):
                for sequence in loader:
                    batch = normalize_data_dp(opt, dtype, sequence)
                    yield batch

        self.training_batch_generator = get_batch(train_loader)
        if opt.inference:
            self.testing_batch_generator = get_dump_batch(test_loader)
        else:
            self.testing_batch_generator = get_batch(test_loader)

    def init_colorize(self):
        if self.opt.dataset in ['KITTI_64', 'KITTI_128', 'KITTI_256', 'Cityscapes_128x256']:
            self.opt.n_class = n_class = 19
            self.pallette = return_colormap('KITTI').byte().numpy().reshape(-1).tolist()
            self.colorize = Colorize(n_class, return_colormap('KITTI'))
        elif self.opt.dataset in ['Pose_64', 'Pose_128']:
            self.opt.n_class = n_class = 25
            self.pallette = return_colormap(N=25).byte().numpy().reshape(-1).tolist()
            self.colorize = Colorize(n_class, return_colormap(N=25))
        else:
            raise ValueError()

    def load_states(self, idx=None):
        if self.opt.deepspeed and not self.opt.load_dp_ckpt:
            if idx is None:
                idx = 'last'
            savedir = self.ckpt_dir / str(idx)
            if savedir is not None:
                try:
                    _, _ = self.encoder.load_checkpoint(savedir, 'encoder')
                    _, _ = self.decoder.load_checkpoint(savedir, 'decoder')
                    _, _ = self.frame_predictor.load_checkpoint(savedir, 'frame_predictor')
                    _, _ = self.posterior.load_checkpoint(savedir, 'posterior')
                    _, _ = self.prior.load_checkpoint(savedir, 'prior')
                    self.global_iter = _['step']
                except:
                    printstr = 'ckpt is not found at: %s' % savedir
                    print_rank_0(printstr)
                    return
                else:
                    printstr = 'ckpt is loaded from: %s' % savedir
                    print_rank_0(printstr)

        if not self.opt.deepspeed and not self.opt.load_ds_ckpt:
            idx = 'last.pth' if idx is None else '%d.pth' % idx
            path = self.ckpt_dir / idx
            try:
                ckpt = torch.load(path)

                self.global_iter = ckpt['global_iter']

                self.frame_predictor.load_state_dict(ckpt['frame_predictor'])
                self.posterior.load_state_dict(ckpt['posterior'])
                self.prior.load_state_dict(ckpt['prior'])
                self.encoder.load_state_dict(ckpt['encoder'])
                self.decoder.load_state_dict(ckpt['decoder'])
            except:
                printstr = 'failed to load ckpt from: %s' % path
                print(printstr)
            else:
                printstr = 'ckpt is loaded from: %s' % path
                print(printstr)

    def dump_states(self, idx=None):
        if self.opt.deepspeed:
            if idx is None:
                idx = 'last'
            savedir = self.ckpt_dir / str(idx)
            client_state = {'step':self.global_iter, 'opt':self.opt}
            self.encoder.save_checkpoint(savedir, 'encoder', client_state)
            self.decoder.save_checkpoint(savedir, 'decoder', client_state)
            self.frame_predictor.save_checkpoint(savedir, 'frame_predictor', client_state)
            self.posterior.save_checkpoint(savedir, 'posterior', client_state)
            self.prior.save_checkpoint(savedir, 'prior', client_state)
        else:
            torch.save({
                'global_iter': self.global_iter,
                'encoder': self.encoder.state_dict(),
                'encoder_optimizer': self.encoder_optimizer.state_dict(),
                'decoder': self.decoder.state_dict(),
                'decoder_optimizer': self.decoder_optimizer.state_dict(),
                'frame_predictor': self.frame_predictor.state_dict(),
                'frame_predictor_optimizer': self.frame_predictor_optimizer.state_dict(),
                'posterior': self.posterior.state_dict(),
                'posterior_optimizer': self.posterior_optimizer.state_dict(),
                'prior': self.prior.state_dict(),
                'prior_optimizer': self.prior_optimizer.state_dict(),
                'opt': self.opt},
                '%s/%s.pth' % (self.ckpt_dir, idx))

    def load_dp_ckpt(self, idx=None):
        idx = 'last.pth' if idx is None else '%d.pth' % idx
        path = self.ckpt_dir / idx
        try:
            ckpt = torch.load(path)
        except FileNotFoundError as e:
            print(e)
            pass
        else:
            self.global_iter = ckpt['global_iter']

            self.encoder = DataParallel(self.encoder)
            self.decoder = DataParallel(self.decoder)
            self.frame_predictor = DataParallel(self.frame_predictor)
            self.posterior = DataParallel(self.posterior)
            self.prior = DataParallel(self.prior)

            self.frame_predictor.load_state_dict(ckpt['frame_predictor'])
            self.posterior.load_state_dict(ckpt['posterior'])
            self.prior.load_state_dict(ckpt['prior'])
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])

            self.encoder = self.encoder.module
            self.decoder = self.decoder.module
            self.frame_predictor = self.frame_predictor.module
            self.posterior = self.posterior.module
            self.prior = self.prior.module

            printstr = 'ckpt is loaded from: %s' % path
            print(printstr)

    def load_ds_ckpt(self, idx=None):
        idx = 'last' if idx is None else str(idx)
        path = str(self.ckpt_dir / idx / '%s/mp_rank_00_model_states.pt')

        try:
            encoder_ckpt = torch.load(path % 'encoder')
            decoder_ckpt = torch.load(path % 'decoder')
            frame_predictor_ckpt = torch.load(path % 'frame_predictor')
            posterior_ckpt = torch.load(path % 'posterior')
            prior_ckpt = torch.load(path % 'prior')
        except FileNotFoundError as e:
            print(e)
            pass
        else:
            self.encoder.load_state_dict(encoder_ckpt['module'])
            self.decoder.load_state_dict(decoder_ckpt['module'])
            self.frame_predictor.load_state_dict(frame_predictor_ckpt['module'])
            self.posterior.load_state_dict(posterior_ckpt['module'])
            self.prior.load_state_dict(prior_ckpt['module'])

            self.encoder_optimizer.load_state_dict(encoder_ckpt['optimizer'])
            self.decoder_optimizer.load_state_dict(decoder_ckpt['optimizer'])
            self.frame_predictor_optimizer.load_state_dict(frame_predictor_ckpt['optimizer'])
            self.posterior_optimizer.load_state_dict(posterior_ckpt['optimizer'])
            self.prior_optimizer.load_state_dict(prior_ckpt['optimizer'])
            self.global_iter = encoder_ckpt['step']
            printstr = 'ckpt is loaded from: %s' % path
            print(printstr)

    def init_loss_functions(self):
        self.kl_criterion = kl_criterion
        self.nll = nn.NLLLoss()

    def train(self, x):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()

        kld = 0
        nll = 0
        prior_hidden = None
        posterior_hidden = None
        frame_predictor_hidden = None
        for i in range(1, self.opt.n_past+self.opt.n_future):
            x_in = x[i-1]
            x_target = x[i]

            h = self.encoder(x_in)
            h_target = self.encoder(x_target)[0]

            if self.opt.last_frame_skip or i < self.opt.n_past + 1:
                h, skip = h
            else:
                h = h[0]

            z_t, mu, logvar, posterior_hidden = self.posterior(h_target, posterior_hidden)
            _, mu_p, logvar_p, prior_hidden = self.prior(h, prior_hidden)
            h_pred, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)

            x_pred = self.decoder([h_pred, skip])
            nll += self.nll(x_pred, x_target.squeeze(1).long())
            kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)

        loss = nll + kld*self.opt.beta
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()

        output = dict()
        normalizer = self.opt.n_past + self.opt.n_future
        output['nll'] = nll.item()/normalizer
        output['kld'] = kld.item()/normalizer

        return output

    @torch.no_grad()
    def validate(self, x):
        kld = 0
        nll = 0
        prior_hidden = None
        posterior_hidden = None
        frame_predictor_hidden = None
        for i in range(1, self.opt.n_past+self.opt.n_future):
            x_in = x[i-1]
            x_target = x[i]

            h = self.encoder(x_in)
            h_target = self.encoder(x_target)[0]

            if self.opt.last_frame_skip or i < self.opt.n_past + 1:
                h, skip = h
            else:
                h = h[0]

            z_t, mu, logvar, posterior_hidden = self.posterior(h_target, posterior_hidden)
            _, mu_p, logvar_p, prior_hidden = self.prior(h, prior_hidden)
            h_pred, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)

            x_pred = self.decoder([h_pred, skip])
            nll += self.nll(x_pred, x_target.squeeze(1).long())
            kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)

        output = dict()
        normalizer = self.opt.n_past + self.opt.n_future
        output['nll'] = nll.item()/normalizer
        output['kld'] = kld.item()/normalizer

        return output

    def solve(self):
        pbar = tqdm(range(self.global_iter, self.opt.max_iter))
        start_time = time.time()
        for _ in pbar:
            self.global_iter += 1

            self.frame_predictor.train()
            self.posterior.train()
            self.prior.train()
            self.encoder.train()
            self.decoder.train()

            x, _ = next(self.training_batch_generator)

            # train
            output = self.train(x)
            nll = output['nll']
            kld = output['kld']

            if self.global_iter % self.opt.log_ckpt_iter == 0:
                # save the model
                self.dump_states(self.global_iter)
                self.dump_states('last')

            if time.time() - start_time > self.opt.log_ckpt_sec:
                # save the model
                self.dump_states('last')
                start_time = time.time()

            if self.global_iter % self.opt.print_iter == 0:
                printstr = '[%02d] nll: %.5f | kld loss: %.5f' % (
                        self.global_iter, nll, kld,)
                #tprint_rank_0(pbar, printstr)
                pbar.set_description(printstr)

            if self.global_iter % self.opt.log_line_iter == 0:
                self.writer.add_scalar('train_nll', nll, global_step=self.global_iter)
                self.writer.add_scalar('train_kld', kld, global_step=self.global_iter)

            if self.global_iter % self.opt.log_img_iter == 0:
                # plot some stuff
                self.frame_predictor.eval()
                self.posterior.eval()
                self.prior.eval()
                self.encoder.eval()
                self.decoder.eval()

                x, _ = next(self.testing_batch_generator)
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        plot(x, self)
                else:
                    plot(x, self)

            if self.global_iter % self.opt.validate_iter == 0:
                nll = 0
                kld = 0
                nvalsample = 0
                for _ in range(100):
                    x, _ = next(self.testing_batch_generator)
                    output = self.validate(x)
                    nll += output['nll']
                    kld += output['kld']
                    nvalsample += x[0].size(0)

                nll /= nvalsample
                kld /= nvalsample
                self.writer.add_scalar('test_nll', nll, global_step=self.global_iter)
                self.writer.add_scalar('test_kld', kld, global_step=self.global_iter)
        pbar.close()

    @torch.no_grad()
    def inference(self):
        topil = transforms.ToPILImage()

        n_prediction = self.opt.n_prediction

        self.frame_predictor.eval()
        self.posterior.eval()
        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()
        for batch_idx, (x_seqs, paths) in tqdm(enumerate(self.testing_batch_generator)):

            # When unrolling step is beyond the number of grund-truth data
            for _ in range(self.opt.n_past + self.opt.n_eval - len(x_seqs)):
                x_seqs.append(x_seqs[-1])
                path_parts = paths[-1][0].split('/')
                name = path_parts[-1]
                if 'KITTI' in self.opt.dataset:
                    newname = '%s_%010d.png' % ('_'.join(name.strip('.png').split('_')[:-1]), int(name.strip('.png').split('_')[-1])+self.opt.frame_sampling_rate)
                elif 'Cityscapes' in self.opt.dataset:
                    new_idx = '%06d' % (int(name.split('_')[-2]) + self.opt.frame_sampling_rate)
                    parts = name.split('_')
                    parts[-2] = new_idx
                    newname = '_'.join(parts)
                elif 'Pose' in self.opt.dataset:
                    newname = newname = 'frame%06d_IUV.png' % (int(name.strip('.png').strip('frame').split('_')[0])+self.opt.frame_sampling_rate)
                newpath = ['/'.join(path_parts[:-1] + [newname])]
                paths.append(newpath)

            x_pred_seqs = []
            for s in range(n_prediction):
                skip = None
                prior_hidden = None
                posterior_hidden = None
                frame_predictor_hidden = None
                x_in = x_seqs[0]
                x_pred_seq = [x_in.data.cpu().byte()]
                for i in range(1, self.opt.n_past+self.opt.n_eval):

                    h = self.encoder(x_in)
                    if self.opt.last_frame_skip or i < self.opt.n_past + 1:
                        h, skip = h
                    else:
                        h = h[0]

                    if i < self.opt.n_past:
                        x_target = x_seqs[i]
                        h_target = self.encoder(x_target)[0]
                        z_t, _, _, posterior_hidden = self.posterior(h_target, posterior_hidden)
                        _, _, _, prior_hidden = self.prior(h, prior_hidden)
                        _, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
                        x_in = x_target
                    else:
                        z_t, _, _, prior_hidden = self.prior(h, prior_hidden)
                        h_pred, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
                        x_in = self.decoder([h_pred, skip]).argmax(dim=1, keepdim=True)

                    x_pred_seq.append(x_in.data.cpu().byte())

                x_pred_seq = torch.stack(x_pred_seq, dim=1)
                x_pred_seqs.append(x_pred_seq)

            x_seqs = torch.cat(x_seqs).data.cpu().byte()            # (n_past+n_eval, 1, H, W)
            x_pred_seqs = torch.cat(x_pred_seqs).transpose(0, 1)    # (n_past+n_eval, n_prediction, 1, H, W)

            for x_gt, x_preds, path in zip(x_seqs, x_pred_seqs, paths):
                path = Path(path[0])
                if 'KITTI' in self.opt.dataset: 
                    maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (0), Path(path.parts[3], path.name))
                elif 'Cityscapes' in self.opt.dataset: 
                    maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (0), Path(path.parts[3], path.name))
                elif 'Pose' in self.opt.dataset:
                    vidname, clipname = path.parts[-3:-1]
                    maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (0), vidname+'_'+clipname, path.name)

                maskpath.parent.mkdir(exist_ok=True, parents=True)
                x_gt = topil(x_gt).convert('P', colors=self.opt.n_class)
                x_gt.putpalette(self.pallette)
                x_gt.save(maskpath)

                for num_x_pred, x_pred in enumerate(x_preds):
                    if 'KITTI' in self.opt.dataset: 
                        maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (num_x_pred+1), Path(path.parts[3], path.name))
                    elif 'Cityscapes' in self.opt.dataset: 
                        maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (num_x_pred+1), Path(path.parts[3], path.name))
                    elif 'Pose' in self.opt.dataset:
                        maskpath = self.preddump_dir.joinpath(str(self.global_iter), 'batch_%05d' % (batch_idx+1), 'sample_%05d' % (num_x_pred+1), vidname+'_'+clipname, path.name)
                    maskpath.parent.mkdir(exist_ok=True, parents=True)
                    x_pred = topil(x_pred).convert('P', colors=self.opt.n_class)
                    x_pred.putpalette(self.pallette)
                    x_pred.save(maskpath)
