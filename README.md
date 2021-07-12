# DCANet
# Datasets
Download the datasets `ShanghaiTech A`, `ShanghaiTech B` and `UCF-QNRF`
Then generate the density maps via `generate_density_map_perfect_names_SHAB_QNRF_NWPU_JHU.py`.
After that, create a folder named `JSTL_large`, and put all the processed data in.

Download the pretrained hrnet model `HRNet-W40-C` from the link `https://github.com/HRNet/HRNet-Image-Classification`.
%
After doing that, download the pretrained model via
```bash
bash download_models.sh
```

# Test
```python
python test.py --domain_center_model='average_clip_domain_center_54.97'
```