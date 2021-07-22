# DCANet
# Datasets
Download the datasets `ShanghaiTech A`, `ShanghaiTech B` and `UCF-QNRF`
Then generate the density maps via `generate_density_map_perfect_names_SHAB_QNRF_NWPU_JHU.py`.
After that, create a folder named `JSTL_large`, and put all the processed data in.

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
