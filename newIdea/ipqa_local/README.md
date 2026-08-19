# ipqa_local —— 可独立运行的 IP-IQA + 局部失真分支工程

这个文件夹是从官方 `IP-IQA` 完整拷贝出来的**独立可运行工程**，并在其中集成了
缝合模块（`ipiqa/models/local_branch/`）。原始源码仓库
`../code/IP-IQA` **保持原样未改动**，服务器上只需携带本文件夹即可。

## 目录结构

```
ipqa_local/
├── clip/                          # 官方仓库内置的 OpenAI CLIP（RN50 等）
├── ipiqa/
│   ├── common/ datasets/ processors/ tasks/ configs/
│   ├── models/
│   │   ├── ipiqa.py               # 已集成 GatedLocalFusion（配置开关驱动）
│   │   ├── utils.py
│   │   └── local_branch/          # 新增缝合模块（自包含，仅依赖 torch）
│   └── projects/agiqa3k/
│       ├── ipiqa.yaml             # 官方 100 epoch 完整配置（原样）
│       ├── ipiqa_quick.yaml       # E0: baseline（20 epoch quick）
│       ├── ipiqa_ours.yaml        # E2: Full Ours
│       └── ipiqa_ours_noattn.yaml # E1: +Weighted Local（无 attention）
├── ../data/                       # 训练数据（在 ipqa_local 上级 newIdea/data/，git 忽略）
│   ├── ckpt/clip/openai/resnet/RN50.pt   # CLIP RN50 权重
│   └── aigc_qa_3k/
│       ├── AGIQA-3K/                     # 2982 张图片
│       ├── data.csv                      # 官方标注
│       └── mos_joint.xlsx                # IP-IQA 需要的标注格式
├── trainer.py
├── train_agiqa3k.py
├── prepare_data.py               # 服务器端用 data.csv 生成 mos_joint.xlsx
├── smoke_test.py                 # 集成正确性验证
├── requirements.txt / setup.py
└── README.md
```

## 服务器环境

```bash
conda create -n ipiqa python=3.10 -y
conda activate ipiqa
# 推荐现代 torch（兼容 RTX 40 系），不要用官方锁死的 torch==2.0.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

> 官方 `requirements.txt` 里钉死 `torch==2.0.1/torchvision==0.15.2`，在 Ada 架构
> （RTX 40 系）和 Python 3.10+ 上不可靠，故用上面的现代版本；其余依赖直接装。
> `pip install -e . --no-deps` 只注册 `ipiqa` 包，避免重复拉 torch。

## 数据准备（如果服务器上没有 ../data）

训练数据统一放在 `ipqa_local` 上级的 `../data/`（即 `newIdea/data/`）：

```bash
# 1) 图片解压到 ../data/aigc_qa_3k/AGIQA-3K/
# 2) 官方 data.csv 放到 ../data/aigc_qa_3k/
python prepare_data.py   # 用 ../data/aigc_qa_3k/data.csv 生成 mos_joint.xlsx
```

## 运行实验（同一 seed 42，单次 80/20 hold-out）

```bash
# E0: baseline
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_quick.yaml --seed 42 --num_cv 1

# E1: +Weighted Local（无 attention）
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ours_noattn.yaml --seed 42 --num_cv 1

# E2: Full Ours
python train_agiqa3k.py --cfg-path ipiqa/projects/agiqa3k/ipiqa_ours.yaml --seed 42 --num_cv 1
```

记录三个实验的 `qual_SROCC / qual_PLCC / qual_KROCC / align_SROCC`。
**最重要的对比是 E2 vs E0（同一划分、同一 seed）**，即计划文档里的
`Y - X`，而不是和论文报告数字直接横比。

## 运行结果（run/ 约定）

每次实验的日志、模型权重、指标统一保存到 `../run/<job_id>_<tag>/`
（即工程根目录 `newIdea/run/`）：

```
newIdea/run/
└── 20260819200_E0_baseline/       # job_id_时间戳 + 实验 tag
    ├── log.txt                    # 每 epoch 训练/验证日志（含全部指标，入库 git）
    ├── checkpoint_best.pth        # 最佳权重（git 忽略，仅本地）
    ├── checkpoint_latest.pth      # 最新权重，随训练覆盖更新（git 忽略，仅本地）
    └── result/                    # 指标输出
```

- `trainer.py` 依据各配置的 `output_dir: "../run"` + `tag` 自动生成带时间戳的子目录。
- 各配置 tag：E0=`E0_baseline`、E1=`E1_weighted`、E2=`E2_full`、
  官方 100ep=`official_100ep`、官方 100ep@224=`official_100ep_224`。
- 权重只保留 `checkpoint_best.pth`（最佳）和 `checkpoint_latest.pth`（最新），不再按
  `save_freq` 保留多个分 epoch 文件；`save_freq` 仅决定最新权重的刷新频率。
- `*.pth` 与 `data/` 体积大，已被 `.gitignore` 排除，不会推送到 git；`log.txt` / `result/` 正常入库。
- 已跑实验结果汇总见 `../RESULTS.md`。

## 集成方式说明

`ipiqa/models/ipiqa.py` 中的改动（与官方版本的最小差异）：

- `forward` 中：`base_output = self.head(global_feat)`，若 `self.local_fusion`
  非空则 `return self.local_fusion(base_output, feat)`，否则原样返回。
- `GatedLocalFusion` 只融合 quality 通道，alignment 通道原样透传。
- 全部由模型配置开关驱动：

| 配置项 | E0 | E1 | E2 |
|---|---|---|---|
| `use_local_branch` | False | True | True |
| `local_use_attention` | - | False | True |
| `local_hidden_dim` | 256 | 256 | 256 |
| `local_gate_init` | -2.0 | -2.0 | -2.0 |

## 显存 / 速度提示

- 服务器若为 3090(24GB)：`batch_size 8 + amp True` 即可，可逐步加。
- 若 8GB 卡（如 RTX 4060 Laptop）：先跑 `batch_size 4 + accum_grad_iters 2`，
  或把 `local_hidden_dim` 降到 128（计划文档第 16 节回退方案）。

## 校验集成正确性

```bash
python smoke_test.py ipiqa/projects/agiqa3k/ipiqa_quick.yaml   # baseline
python smoke_test.py ipiqa/projects/agiqa3k/ipiqa_ours.yaml     # full ours
```
