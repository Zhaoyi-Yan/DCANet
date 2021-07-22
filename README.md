# DCANet
# Datasets Preparation
Download the datasets `ShanghaiTech A`, `ShanghaiTech B` and `UCF-QNRF`
Then generate the density maps via `generate_density_map_perfect_names_SHAB_QNRF_NWPU_JHU.py`.
After that, create a folder named `JSTL_large_dataset`, and directly copy all the processed data in `JSTL_large_dataset`.

The tree of the folder should be:
```bash
`DATASET` is `SHA`, `SHB` or `QNRF_large`.

-JSTL_large_dataset
   -den
       -test
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
       -train
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
   -ori
       -test_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
       -train_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
```

Download the pretrained hrnet model `HRNet-W40-C` from the link `https://github.com/HRNet/HRNet-Image-Classification` and put it directly in the root path of the repository.
%
After doing that, download the pretrained model via
```bash
bash download_models.sh
```
And put the model into folder './output', change the model name in `test.sh` or `test_fast.sh` scripts.

# Test
```bash
sh test.sh
```
Or if you have two GPUs, then
```bash
sh test_fast.sh
```
