import util.utils as util
import os
import torch

class config(object):
    def __init__(self, opt):
        self.opt = opt
        self.unknown_folder = opt.unknown_folder
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

        if self.dataset_name == "SHA":
            assert 1==2
            self.eval_num = 182
            self.train_num = 300

            self.train_gt_map_path = "YZ_crowd_dataset/SH_part_A_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/SH_part_A_yz/test"
            self.train_img_path = "Crowd_dataset/ShanghaiTech/part_A_final/train_data/images"
            self.eval_img_path = "Crowd_dataset/ShanghaiTech/part_A_final/test_data/images"
            self.eval_gt_path = "Crowd_dataset/ShanghaiTech/part_A_final/test_data/ground_truth"

        elif self.dataset_name == "SHB":
            assert 1==2
            self.eval_num = 316
            self.train_num = 400

            self.train_gt_map_path = "YZ_crowd_dataset/SH_part_B_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/SH_part_B_yz/test"
            self.train_img_path = "Crowd_dataset/ShanghaiTech/part_B_final/train_data/images"
            self.eval_img_path = "Crowd_dataset/ShanghaiTech/part_B_final/test_data/images"
            self.eval_gt_path = "Crowd_dataset/ShanghaiTech/part_B_final/test_data/ground_truth"

        elif self.dataset_name == "QNRF":
            assert 1==2
            self.eval_num = 334
            self.train_num = 1201

            self.train_gt_map_path = "YZ_crowd_dataset/QNRF_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/QNRF_yz/test"
            self.train_img_path = "Crowd_dataset/UCF-QNRF_ECCV18/train"
            self.eval_img_path = "Crowd_dataset/UCF-QNRF_ECCV18/test"
            # As QNRF the ann.mat is in the same folder of images
            self.eval_gt_path = "Crowd_dataset/UCF-QNRF_ECCV18/test"
        # for this repo, just support JSTL
        elif self.dataset_name == 'JSTL':
            self.eval_num = 832
            self.train_num = 1901

            self.train_gt_map_path = 'JSTL_dataset/den/train'
            self.eval_gt_map_path = 'JSTL_dataset/den/test'
            self.train_img_path = 'JSTL_dataset/ori/train_data/images'
            self.eval_img_path = 'JSTL_dataset/ori/test_data/images'
            self.eval_gt_path = 'JSTL_dataset/ori/test_data/ground_truth'

        elif self.dataset_name == 'JSTL_large':

            self.train_gt_map_path = 'JSTL_large_dataset/den/train'
            self.eval_gt_map_path = 'JSTL_large_dataset/den/test'
            self.train_img_path = 'JSTL_large_dataset/ori/train_data/images'
            self.eval_img_path = 'JSTL_large_dataset/ori/test_data/images'
            self.eval_gt_path = 'JSTL_large_dataset/ori/test_data/ground_truth'
            self.datasets_com = ['SHA', 'SHB', 'QNRF_large']

        # for extra images contain no gt counts
        elif self.dataset_name == 'unknown_img':
            self.eval_num = 1
            self.train_num = 1 # useless in fact

            self.train_gt_map_path = 'x'
            self.eval_gt_map_path = 'x'
            self.train_img_path = 'x'
            self.eval_img_path = self.unknown_folder
            self.eval_gt_path = 'x'

        elif self.dataset_name == 'JSTL_SH_LgQF_NU_JU_oriImg':
      #      self.eval_num = 832
      #      self.train_num = 1901

            self.train_gt_map_path = 'JSTL_SH_LgQF_NU_JU_oriImg/den/train'
            self.eval_gt_map_path = 'JSTL_SH_LgQF_NU_JU_oriImg/den/test'
            self.train_img_path = 'JSTL_SH_LgQF_NU_JU_oriImg/ori/train_data/images'
            self.eval_img_path = 'JSTL_SH_LgQF_NU_JU_oriImg/ori/test_data/images'
            self.eval_gt_path = 'JSTL_SH_LgQF_NU_JU_oriImg/ori/test_data/ground_truth'

            self.datasets_com = ['SHA', 'SHB', 'QNRF_large', 'NWPU_large', 'JHU_large']

        elif self.dataset_name == 'JSTL_SH_LgQF_NU_JU_BD':
      #      self.eval_num = 832
      #      self.train_num = 1901

            self.train_gt_map_path = 'JSTL_SH_LgQF_NU_JU_BD/den/train'
            self.eval_gt_map_path = 'JSTL_SH_LgQF_NU_JU_BD/den/test'
            self.train_img_path = 'JSTL_SH_LgQF_NU_JU_BD/ori/train_data/images'
            self.eval_img_path = 'JSTL_SH_LgQF_NU_JU_BD/ori/test_data/images'
            self.eval_gt_path = 'JSTL_SH_LgQF_NU_JU_BD/ori/test_data/ground_truth'

            self.datasets_com = ['SHA', 'SHB', 'QNRF_large', 'NWPU_large', 'JHU_large', 'BDdata_large']


        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
