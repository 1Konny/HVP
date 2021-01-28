import imageio
import numpy as np

import torch
from torch.autograd import Variable


class DummyDataset():
    def __init__(self, N=10000):
        self.len = N
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return torch.zeros(1)


def load_dataset(opt):
    if opt.dataset in ['KITTI_64', 'KITTI_256']: 
        from data.kitti import KITTI, KITTITest
        if not opt.inference:
            train_data = KITTI(
                    data_root=opt.data_root,
                    split='train',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_future,
                    hflip=True,
                    )
            test_data = KITTI(
                    data_root=opt.data_root,
                    split='val',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
        else:
            train_data = DummyDataset()
            test_data = KITTITest(
                    data_root=opt.data_root,
                    split='test',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
    elif opt.dataset in ['Cityscapes_128x256']:
        from data.cityscapes import Cityscapes, CityscapesTest
        if not opt.inference:
            train_data = Cityscapes(
                    data_root=opt.data_root,
                    split='train',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_future,
                    hflip=True,
                    )
            test_data = CityscapesTest(
                    data_root=opt.data_root,
                    split='val',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
        else:
            train_data = DummyDataset()
            test_data = CityscapesTest(
                    data_root=opt.data_root,
                    split='test',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
    elif opt.dataset in ['Pose_64', 'Pose_128']:
        from data.pose import Pose, PoseTest 
        if not opt.inference:
            train_data = Pose(
                    data_root=opt.data_root,
                    split='train',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_future,
                    hflip=True,
                    )
            test_data = PoseTest(
                    data_root=opt.data_root,
                    split='test',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
        else:
            train_data = DummyDataset()
            test_data = PoseTest(
                    data_root=opt.data_root,
                    split='test',
                    frame_sampling_rate=opt.frame_sampling_rate,
                    video_length=opt.n_past+opt.n_eval,
                    hflip=False,
                    )
    
    return train_data, test_data


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]


def normalize_data_dp(opt, dtype, sequence):
    data, data_path = sequence
    if opt.dataset == 'smmnist' or opt.dataset == 'kth' or opt.dataset == 'bair' :
        data.transpose_(0, 1)
        data.transpose_(3, 4).transpose_(2, 3)
    else:
        data.transpose_(0, 1)

    return sequence_input(data, dtype), data_path


@torch.no_grad()
def plot(x, self, n_sample_per_video=5, max_num_video=4):
    opt = self.opt
    x = [x_ for x_ in x]
    if opt.n_past+opt.n_eval > len(x): 
        for _ in range(opt.n_past+opt.n_eval - len(x)):
            x.append(x[-1])
    gen_seq = [[] for i in range(n_sample_per_video)]
    gt_seq = [x[i] for i in range(len(x))]
    recon_seq = [x[i] for i in range(len(x))]

    # get reconstruction
    frame_predictor_hidden = None
    posterior_hidden = None
    prior_hidden = None

    x_in = x[0]
    batch_size = x_in.shape[0]
    for i in range(1, opt.n_eval):
        h = self.encoder(x_in)
        if opt.last_frame_skip or i < opt.n_past + 1:
            h, skip = h
        else:
            h, _ = h

        h_target = self.encoder(x[i])
        h_target = h_target[0]

        _, z_t, _, posterior_hidden = self.posterior(h_target, posterior_hidden)
        h, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
        x_pred = self.decoder([h, skip]).argmax(dim=1, keepdim=True)
        recon_seq[i] = x_pred
        x_in = x[i]

    # get prediction
    for s in range(n_sample_per_video):
        frame_predictor_hidden = None
        posterior_hidden = None
        prior_hidden = None

        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = self.encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past + 1:
                h, skip = h
            else:
                h, _ = h
            if i < opt.n_past:
                h_target = self.encoder(x[i])
                h_target = h_target[0]
                z_t, _, _, posterior_hidden = self.posterior(h_target, posterior_hidden)
                _, _, _, prior_hidden = self.prior(h, prior_hidden)
                _, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
                x_in = x[i]
            else:
                z_t, _, _, prior_hidden = self.prior(h, prior_hidden)
                h, frame_predictor_hidden = self.frame_predictor(torch.cat([h, z_t], 1), frame_predictor_hidden)
                x_in = self.decoder([h, skip]).argmax(dim=1, keepdim=True)

            gen_seq[s].append(x_in)

    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(batch_size, max_num_video)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])

        s_list = list(range(n_sample_per_video))
        for ss in range(len(s_list)):
            s = s_list[ss]
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
        for t in range(opt.n_eval):
            n_pad = 2
            if t < opt.n_past:
                canvas_color = green = [0., 1., 0.] # green
            elif opt.n_past <= t < opt.n_past+opt.n_future:
                canvas_color = yellow = [1., 1., 0.] # yellow 
            else:
                canvas_color = red = [1., 0., 0.] # red

            row = []

            gen = self.colorize(gt_seq[t][i].unsqueeze(0)).float().div(255)
            h, w = gen.shape[-2:]
            canvas = torch.tensor(green, device=gen.device, dtype=gen.dtype).view(3, 1, 1).repeat(1, h+n_pad*2, w+n_pad*2)
            canvas[:, n_pad:n_pad+h, n_pad:n_pad+w] = gen
            row.append(canvas)

            recon = self.colorize(recon_seq[t][i].unsqueeze(0)).float().div(255)
            h, w = recon.shape[-2:]
            canvas_recon = torch.tensor(green, device=recon.device, dtype=recon.dtype).view(3, 1, 1).repeat(1, h+n_pad*2, w+n_pad*2)
            canvas_recon[:, n_pad:n_pad+h, n_pad:n_pad+w] = recon
            row.append(canvas_recon)

            for ss in range(len(s_list)):
                s = s_list[ss]

                gen = self.colorize(gen_seq[s][t][i].unsqueeze(0)).float().div(255)
                h, w = gen.shape[-2:]
                canvas = torch.tensor(canvas_color, device=gen.device, dtype=gen.dtype).view(3, 1, 1).repeat(1, h+n_pad*2, w+n_pad*2)
                canvas[:, n_pad:n_pad+h, n_pad:n_pad+w] = gen
                row.append(canvas)

            gifs[t].append(row)

    fname = '%s/sample_%d.gif' % (self.sample_dir, self.global_iter)
    save_gif(fname, gifs)

    result = torch.from_numpy(np.array(imageio.mimread(fname, memtest=False))).transpose(2, 3).transpose(1, 2)
    self.writer.add_video('video_pred', result.unsqueeze(0), self.global_iter)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)


        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)


        return result


    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)


        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).mul(255)
        images.append(img.numpy().astype(np.uint8))
    imageio.mimsave(filename, images, duration=duration)


def return_colormap(dataset=None, N=None):
    if dataset == 'KITTI':
        color_map = torch.tensor([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ])
    elif dataset is None and N is not None:
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        color_map = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            color_map[i, 0] = r
            color_map[i, 1] = g
            color_map[i, 2] = b
        color_map = torch.from_numpy(color_map)
    return color_map


class Colorize(object):
    def __init__(self, n=35, cmap=None):
        if cmap is None:
            raise NotImplementedError()
            self.cmap = labelcolormap(n)
        else:
            self.cmap = cmap
        self.cmap = self.cmap[:n]

    def preprocess(self, x):
        if len(x.size()) > 3 and x.size(1) > 1:
            # if x has a shape of [B, C, H, W],
            # where B and C denote a batch size and the number of semantic classes,
            # then translate it into a shape of [B, 1, H, W]
            x = x.argmax(dim=1, keepdim=True).float()
        assert (len(x.shape) == 4) and (x.size(1) == 1), 'x should have a shape of [B, 1, H, W]'
        return x

    def __call__(self, x):
        x = self.preprocess(x)
        if (x.dtype == torch.float) and (x.max() < 2):
            x = x.mul(255).long()

        color_images = []
        gray_image_shape = x.shape[1:]
        for gray_image in x:
            color_image = torch.ByteTensor(3, *gray_image_shape[1:]).fill_(0)
            for label, cmap in enumerate(self.cmap):
                mask = (label == gray_image[0]).cpu()
                color_image[0][mask] = cmap[0]
                color_image[1][mask] = cmap[1]
                color_image[2][mask] = cmap[2]

            color_images.append(color_image)
        color_images = torch.stack(color_images)
        return color_images
