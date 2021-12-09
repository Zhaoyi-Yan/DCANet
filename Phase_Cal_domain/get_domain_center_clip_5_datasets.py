import numpy as np
import torch
import torch.nn.functional as F
import os
import glob

npys = glob.glob("feature_weights_total_MAE_69.45_scale_0.0/train/*npz")

qnrf_lst = []
sha_lst = []
shb_lst = []
nwpu_lst = []
jhu_lst = []

for npy in npys:
    attn_source = np.load(npy)
    attn_info = attn_source['delta_mae']
    attn_info = torch.from_numpy(attn_info)
    attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3 = torch.chunk(attn_info, 4, dim=0)

    if npy.find("QNRF") != -1:
        qnrf_lst.append([attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3])
    elif npy.find("SHA") != -1:
        sha_lst.append([attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3])
    elif npy.find("SHB") != -1:
        shb_lst.append([attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3])
    elif npy.find("NWPU") != -1:
        nwpu_lst.append([attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3])
    elif npy.find("JHU") != -1:
        jhu_lst.append([attn_info_b0, attn_info_b1, attn_info_b2, attn_info_b3])
    else:
        raise ValueError("...")

print('Overall for dataset: ')
print('SHA: ', str(len(sha_lst)), ' SHB: ', str(len(shb_lst)), ' QNRF: ', str(len(qnrf_lst)), ' NWPU: ', str(len(nwpu_lst)), ' JHU: ', str(len(jhu_lst)))

# calculate the domain center
# for now, directly figure out the average mean(Or we can use the clustering method)

# Method 1: directly calculate mean of each dataset
dMAE_per_c_SHA = torch.zeros(512, 1)
dMAE_per_c_SHB = torch.zeros(512, 1)
dMAE_per_c_QNRF = torch.zeros(512, 1)
dMAE_per_c_NWPU = torch.zeros(512, 1)
dMAE_per_c_JHU = torch.zeros(512, 1)

for k in sha_lst:
    dMAE_per_c_SHA[0:128, :] += k[0]
    dMAE_per_c_SHA[128:128*2, :] += k[1]
    dMAE_per_c_SHA[128*2:128*3, :] += k[2]
    dMAE_per_c_SHA[128*3:128*4, :] += k[3]

dMAE_per_c_SHA /= len(sha_lst)

for k in shb_lst:
    dMAE_per_c_SHB[0:128, :] += k[0]
    dMAE_per_c_SHB[128:128*2, :] += k[1]
    dMAE_per_c_SHB[128*2:128*3, :] += k[2]
    dMAE_per_c_SHB[128*3:128*4, :] += k[3]

dMAE_per_c_SHB /= len(shb_lst)

for k in qnrf_lst:
    dMAE_per_c_QNRF[0:128, :] += k[0]
    dMAE_per_c_QNRF[128:128*2, :] += k[1]
    dMAE_per_c_QNRF[128*2:128*3, :] += k[2]
    dMAE_per_c_QNRF[128*3:128*4, :] += k[3]

dMAE_per_c_QNRF /= len(qnrf_lst)

for k in nwpu_lst:
    dMAE_per_c_NWPU[0:128, :] += k[0]
    dMAE_per_c_NWPU[128:128*2, :] += k[1]
    dMAE_per_c_NWPU[128*2:128*3, :] += k[2]
    dMAE_per_c_NWPU[128*3:128*4, :] += k[3]

dMAE_per_c_NWPU /= len(nwpu_lst)

for k in jhu_lst:
    dMAE_per_c_JHU[0:128, :] += k[0]
    dMAE_per_c_JHU[128:128*2, :] += k[1]
    dMAE_per_c_JHU[128*2:128*3, :] += k[2]
    dMAE_per_c_JHU[128*3:128*4, :] += k[3]

dMAE_per_c_JHU /= len(jhu_lst)

# Get the channels that are harmful to the performance
# For each channel, when removing it, delta_MAE > 0, then the channel is important
# Original MAE=100, after zeroing out the channel, then MAE=110, so delta_MAE > 0,
# It means, the channel is helpful in reduce MAE. So the channel is important
a_un_lst, b_un_lst, q_un_lst, n_un_lst, j_un_lst = [], [], [], [], []
for i in range(128*4):
    if dMAE_per_c_SHA[i, 0].item() < 0:
        print('channel: ', str(i), ' is NOT importance for SHA')
        dMAE_per_c_SHA[i, :] = 0
        a_un_lst.append(i)
    if dMAE_per_c_SHB[i, 0].item() < 0:
        print('channel: ', str(i), ' is NOT importance for SHB')
        dMAE_per_c_SHB[i, 0] = 0
        b_un_lst.append(i)
    if dMAE_per_c_QNRF[i, 0].item() < 0:
        print('channel: ', str(i), ' is NOT importance for QNRF')
        dMAE_per_c_QNRF[i, 0] = 0
        q_un_lst.append(i)
    if dMAE_per_c_NWPU[i, 0].item() < 0:
        print('channel: ', str(i), ' is NOT importance for NWPU')
        dMAE_per_c_NWPU[i, 0] = 0
        n_un_lst.append(i)
    if dMAE_per_c_JHU[i, 0].item() < 0:
        print('channel: ', str(i), ' is NOT importance for JHU')
        dMAE_per_c_JHU[i, 0] = 0
        j_un_lst.append(i)

print('SHA unimportant channels:')
print(len(a_un_lst))
print(a_un_lst)
print('SHB unimportant channels:')
print(len(b_un_lst))
print(b_un_lst)
print('QNRF unimportant channels:')
print(len(q_un_lst))
print(q_un_lst)
print('NWPU unimportant channels:')
print(len(n_un_lst))
print(n_un_lst)
print('JHU unimportant channels:')
print(len(j_un_lst))
print(j_un_lst)

# After zero out the unimportant channels, we softmax them
dMAE_SHA_b0, dMAE_SHA_b1, dMAE_SHA_b2, dMAE_SHA_b3 = torch.chunk(dMAE_per_c_SHA, 4, dim=0)
dMAE_SHB_b0, dMAE_SHB_b1, dMAE_SHB_b2, dMAE_SHB_b3 = torch.chunk(dMAE_per_c_SHB, 4, dim=0)
dMAE_QNRF_b0, dMAE_QNRF_b1, dMAE_QNRF_b2, dMAE_QNRF_b3 = torch.chunk(dMAE_per_c_QNRF, 4, dim=0)
dMAE_NWPU_b0, dMAE_NWPU_b1, dMAE_NWPU_b2, dMAE_NWPU_b3 = torch.chunk(dMAE_per_c_NWPU, 4, dim=0)
dMAE_JHU_b0, dMAE_JHU_b1, dMAE_JHU_b2, dMAE_JHU_b3 = torch.chunk(dMAE_per_c_JHU, 4, dim=0)

attn_SHA_b0 = F.softmax(dMAE_SHA_b0, dim=0)
attn_SHA_b1 = F.softmax(dMAE_SHA_b1, dim=0)
attn_SHA_b2 = F.softmax(dMAE_SHA_b2, dim=0)
attn_SHA_b3 = F.softmax(dMAE_SHA_b3, dim=0)

attn_SHB_b0 = F.softmax(dMAE_SHB_b0, dim=0)
attn_SHB_b1 = F.softmax(dMAE_SHB_b1, dim=0)
attn_SHB_b2 = F.softmax(dMAE_SHB_b2, dim=0)
attn_SHB_b3 = F.softmax(dMAE_SHB_b3, dim=0)

attn_QNRF_b0 = F.softmax(dMAE_QNRF_b0, dim=0)
attn_QNRF_b1 = F.softmax(dMAE_QNRF_b1, dim=0)
attn_QNRF_b2 = F.softmax(dMAE_QNRF_b2, dim=0)
attn_QNRF_b3 = F.softmax(dMAE_QNRF_b3, dim=0)

attn_NWPU_b0 = F.softmax(dMAE_NWPU_b0, dim=0)
attn_NWPU_b1 = F.softmax(dMAE_NWPU_b1, dim=0)
attn_NWPU_b2 = F.softmax(dMAE_NWPU_b2, dim=0)
attn_NWPU_b3 = F.softmax(dMAE_NWPU_b3, dim=0)

attn_JHU_b0 = F.softmax(dMAE_JHU_b0, dim=0)
attn_JHU_b1 = F.softmax(dMAE_JHU_b1, dim=0)
attn_JHU_b2 = F.softmax(dMAE_JHU_b2, dim=0)
attn_JHU_b3 = F.softmax(dMAE_JHU_b3, dim=0)



for i in range(4):
    print('--------- *********** ----------')
    print('for branch ', str(i))
    print('A vs B:')
    print(F.cosine_similarity(eval('attn_SHA_b'+str(i)), eval('attn_SHB_b'+str(i)), dim=0))
    print('B vs Q:')
    print(F.cosine_similarity(eval('attn_SHB_b'+str(i)), eval('attn_QNRF_b'+str(i)), dim=0))
    print('A vs Q:')
    print(F.cosine_similarity(eval('attn_SHA_b'+str(i)), eval('attn_QNRF_b'+str(i)), dim=0))

G_SHA = torch.cat([attn_SHA_b0, attn_SHA_b1, attn_SHA_b2, attn_SHA_b3], dim=0)
G_SHB = torch.cat([attn_SHB_b0, attn_SHB_b1, attn_SHB_b2, attn_SHB_b3], dim=0)
G_QNRF = torch.cat([attn_QNRF_b0, attn_QNRF_b1, attn_QNRF_b2, attn_QNRF_b3], dim=0)
G_NWPU = torch.cat([attn_NWPU_b0, attn_NWPU_b1, attn_NWPU_b2, attn_NWPU_b3], dim=0)
G_JHU = torch.cat([attn_JHU_b0, attn_JHU_b1, attn_JHU_b2, attn_JHU_b3], dim=0)

print('--------For all branches-----------')
print('A vs B')
print(F.cosine_similarity(G_SHA, G_SHB, dim=0))
print('B vs Q')
print(F.cosine_similarity(G_SHB, G_QNRF, dim=0))
print('A vs Q')
print(F.cosine_similarity(G_SHA, G_QNRF, dim=0))

np.savez('average_clip_domain_center_69.45.npz', G_SHA = G_SHA, G_SHB = G_SHB, G_QNRF = G_QNRF, G_NWPU = G_NWPU, G_JHU = G_JHU)



