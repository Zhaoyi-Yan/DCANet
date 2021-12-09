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
        self.scene_index = opt.scene_index # it can be adopted by worldexpo or UCF_CC_50

        print(self.dataset_name)
        if self.dataset_name == "SHA":
            self.eval_num = 182
            self.train_num = 300

            self.train_gt_map_path = "YZ_crowd_dataset/SH_part_A_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/SH_part_A_yz/test"
            self.train_img_path = "Crowd_dataset/ShanghaiTech/part_A_final/train_data/images"
            self.eval_img_path = "Crowd_dataset/ShanghaiTech/part_A_final/test_data/images"
            self.eval_gt_path = "Crowd_dataset/ShanghaiTech/part_A_final/test_data/ground_truth"

        elif self.dataset_name == "SHB":
            self.eval_num = 316
            self.train_num = 400

            self.train_gt_map_path = "YZ_crowd_dataset/SH_part_B_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/SH_part_B_yz/test"
            self.train_img_path = "Crowd_dataset/ShanghaiTech/part_B_final/train_data/images"
            self.eval_img_path = "Crowd_dataset/ShanghaiTech/part_B_final/test_data/images"
            self.eval_gt_path = "Crowd_dataset/ShanghaiTech/part_B_final/test_data/ground_truth"

        elif self.dataset_name == "QNRF":
            self.eval_num = 334
            self.train_num = 1201

            self.train_gt_map_path = "YZ_crowd_dataset/QNRF_yz/train"
            self.eval_gt_map_path = "YZ_crowd_dataset/QNRF_yz/test"
            self.train_img_path = "Crowd_dataset/UCF-QNRF_ECCV18/train"
            self.eval_img_path = "Crowd_dataset/UCF-QNRF_ECCV18/test"
            # As QNRF the ann.mat is in the same folder of images
            self.eval_gt_path = "Crowd_dataset/UCF-QNRF_ECCV18/test"

        elif self.dataset_name == "UCF50":
            self.eval_num = 10
            self.train_num = 0 # just set 0, as we do not use it for training

            self.train_gt_map_path = "UCF50_yz/train" # dummy path
            self.eval_gt_map_path = "UCF50_yz/test/t" + str(self.scene_index)

            self.train_img_path = "UCF_CC_50/images/UCF_CC_50_img" #dummy path
            self.eval_img_path = "UCF_CC_50/images/UCF_CC_50_img/t" + str(self.scene_index)
            self.eval_gt_path = "UCF_CC_50/UCF_CC_50_mat/t" + str(self.scene_index)

        elif self.dataset_name == "WorldExpo": # as there exist masks, so I have to using simialr tesing strategy as UCF50
            self.eval_num = 120 if self.scene_index != 1 else 119 # the first scene only contains 119 anntations.
            self.train_num = 3380

            self.train_gt_map_path = "worldExpo_processed/train"
            self.eval_gt_map_path = "worldExpo_processed/test/" + str(self.scene_index)

            self.train_img_path = "worldexpo/train_frame/"
            self.eval_img_path = "worldexpo/test_frame/" + str(self.scene_index)
            self.eval_gt_path = "worldexpo/test_label/" + str(self.scene_index)



        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
