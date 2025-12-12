# Frame Selector
基于RL范式的帧选择器，用于稀疏化帧以用于下游的3D重建器(VGGT, FastVGGT).


## DATE 2025.12.08
### Done
- Input & Training & Output
  - [x] Input：(S, 3, 392, 518) 的时序图像序列
  - [x] **CLIP Image Encoder** 输出的图像特征 $F_{image}$ (e.g. 1000 x 512)；
  - [x] 基于**K-Means**，将 $F_{image}$ 划分为若干-clusters(e.g. 100个clusters)；
  - [x] for each cluster， do： **LSTM Cell**在cluster内，按时序前向推理 -> logits，并在cluster内做softmax;
  - [x] 训练时，使用**Gumbel Softmax**采样 (没有使用原始的multinomial采样)；
  - [x] 推理时，K-Means -> Cluster内Forward -> cluster内logit最高作为保留索引

- Training Procedure：
  - [x] ground truth：使用**FastVGGT** forward，并体素化(voxel_size=0.01)降采样，得到 $P_{full}$
  - [x] prediction: 使用**FastVGGT**，对各cluster内选出的帧 $f_{sel}$ 执行forward，并体素化(voxel_size=0.01)降采样，得到 $P_{sel}$
  - [x] 运行ICP算法 将 $P_{sel}$ 对齐至 $P_{full}$，使用 $Coverage = \frac{N_{inlier}}{{N_{total}}}$ 作为Reward
  - NAS baseline：
    - [x] 采用EMA baseline：每次的advantage是相较于EMA的结果，收敛速度较慢；
    - [x] 强基线：使用***等步长均匀采样***重建点云$P_{uniform}$，执行ICP并计算Coverage，作为baseline reward，每次advantage计算都是相较于该baseline。收敛速度较快

- Evaluation
    - 整体evaluation框架遵循 **CUT3R** 的实现，但是做了以下修改：
      - [x] 对整个序列图像的ground truth点云执行体素化降采样，得到 $P_{gt}$；
      - [x] 对selector的预测点云执行体素化降采样，得到 $P_{pred}$;
      - [x] 计算**Accurarcy: pred到最近邻gt平均距离**和**Completeness: gt到最近邻pred的平均距离**以及***Chamfer Distance=Acc+Comp***
      - [x] 引入点云生成评估指标 **Coverage**: 预测点云对完整场景的覆盖度

- Results
  - [x] 基于EMA baseline：
  - [x] 基于unifrom baseline：

### TODO
#### Stage 1
- Debug 相关工作
  - [ ] 训练时，可视化 $P_{full}$ 和 $P_{sel}$ 经过ICP后，能否正确对齐；
  - [ ] 评估时，可视化 $P_{gt}$ 和 $P_{pred}$ 经过ICP后，能否正确对齐；

- Training
  - [ ] 参考**图像检索相关工作**，检查图像特征 $F_{image}$ 是否正确**归一化**，以用于后续图像聚类；

- Evaluation
  - [x] **保留现有评估准则**
  - [ ] 引入7Scenes的training dataset

#### Stage 2
- Input & Output
  - [ ] **Highlight**：100 budget过于稠密，考虑减少budget (e.g. 50 or 30)
  - [ ] **Highlight**: 不直接使用完整 $Sequence$ 输入FastVGGT，作为pseudo ground-truth；而是用**簇内的帧**作为输入，从而得到pseudo ground-truth：
    1. 稀疏reward: 每个cluster内选择，遍历所有cluster后，**综合各簇的pseudo pcd，并计算一次reward**
    2. 贪婪reward: 一个sequence作为一个episode，每个step**计算一次reward**，并立即更新controller

- Training
  - [ ] 考虑直接使用embedding编码各图像的位置，结合**可微分聚类**，训练LSTMCell
  - [ ] follow **NeU-NBV**的方法，训练一个相较于当前视角的**信息增益评估器**


## Date 2025.12.xx