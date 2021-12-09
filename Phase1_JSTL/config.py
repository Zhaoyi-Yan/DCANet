import util.utils as util
import os
import torch

class config(object):
    def __init__(self, opt):
        self.opt = opt
        self.min_mae = 10240000
        self.min_loss = 10240000
        self.dataset_name = opt.dataset_name
        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.eval_per_step = opt.eval_per_step
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model_save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name) # path of saving model
        self.epoch = opt.max_epochs
        self.mode = opt.mode
        self.is_random_hsi = opt.is_random_hsi
        self.is_flip = opt.is_flip

        if  self.dataset_name == 'JSTL':

            self.train_gt_map_path = 'JSTL_dataset/den/train'
            self.eval_gt_map_path = 'JSTL_dataset/den/test'
            self.train_img_path = 'JSTL_dataset/ori/train_data/images'
            self.eval_img_path = 'JSTL_dataset/ori/test_data/images'
            self.eval_gt_path = 'JSTL_dataset/ori/test_data/ground_truth'

            self.datasets_com = ['SHA', 'SHB', 'QNRF']

        elif self.dataset_name == 'JSTL_large':

            self.train_gt_map_path = 'JSTL_large_dataset/den/train'
            self.eval_gt_map_path = 'JSTL_large_dataset/den/test'
            self.train_img_path = 'JSTL_large_dataset/ori/train_data/images'
            self.eval_img_path = 'JSTL_large_dataset/ori/test_data/images'
            self.eval_gt_path = 'JSTL_large_dataset/ori/test_data/ground_truth'

            self.datasets_com = ['SHA', 'SHB', 'QNRF_large']

        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
