import os
import clip
import torch.nn as nn
import torch
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.utils import TextAttentionPool2d, interpolate_pos_embed, disabled_train, freeze_module, MLPHead
from ipiqa.models.local_branch import GatedLocalFusion

from ipiqa.common.registry import registry

@registry.register_model("ipiqa")
class IPIQA(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/ipiqa.yaml",
    }
    def __init__(
                self,
                base_ckpt='', # your path for clip resnet (default in `ipiqa.yaml`: ../data/ckpt/clip/openai/resnet/RN50.pt)
                input_resolution=512,
                output_dim=None,
                use_mlp_head=False,
                dropout_rate=0.,
                freeze_text=True,
                head_scale=None,
                qa_token=False,
                use_local_branch=False,
                local_hidden_dim=256,
                local_gate_init=-2.0,
                local_use_attention=True,
                local_branch_type="weighted",
                ms_num_heads=4,
                ms_mlp_ratio=2.0,
                ms_refine_gate_init=-2.0,
                ms_use_dual_attention=True,
                ms_aggregation="weighted",
        ):
        super().__init__()
        clip_ckpt = clip.load(base_ckpt, device="cpu")[0]
        self.resnet50 = clip_ckpt.visual

        self.txt_model = clip_ckpt.transformer
        self.wte = clip_ckpt.token_embedding
        self.ln_final = clip_ckpt.ln_final
        self.txt_pos = clip_ckpt.positional_embedding
        self.text_projection = clip_ckpt.text_projection

        self.dtype = self.resnet50.conv1.weight.dtype

        self.feature_dim = self.resnet50.attnpool.c_proj.out_features
        self.resnet50.attnpool.positional_embedding = nn.Parameter(
                interpolate_pos_embed(self.resnet50.attnpool.positional_embedding,input_resolution=input_resolution))
        self.attnpool = self.resnet50.attnpool
        self.resnet50.attnpool = nn.Identity()
        self.txt_attnpool = TextAttentionPool2d(input_resolution//32,embed_dim=2048,txt_dim=1024,num_heads=32,output_dim=1024)

        if use_mlp_head and output_dim:
            self.head = MLPHead(self.feature_dim*2,output_dim,dropout_rate)
        else:
            self.head = nn.Linear(self.feature_dim*2,output_dim) if output_dim else nn.Identity()

        if freeze_text:
            freeze_module(self.txt_model)
            if not qa_token:
                freeze_module(self.wte)
            else:
                print('use qa-token, unfreeze wte ...')
            freeze_module(self.ln_final)
            freeze_module(self.txt_pos)
            freeze_module(self.text_projection)

        self.head_scale = head_scale

        if use_local_branch:
            self.local_fusion = GatedLocalFusion(
                in_channels=2048,  # CLIP RN50 spatial feature 通道（resnet 末层输出），
                                   # 注意：self.feature_dim(1024) 是 attnpool 输出维，不是这里
                hidden_dim=local_hidden_dim,
                spatial_dim=(input_resolution // 32) ** 2,
                gate_init=local_gate_init,
                use_attention=local_use_attention,
                branch_type=local_branch_type,
                ms_num_heads=ms_num_heads,
                ms_mlp_ratio=ms_mlp_ratio,
                ms_refine_gate_init=ms_refine_gate_init,
                ms_use_dual_attention=ms_use_dual_attention,
                ms_aggregation=ms_aggregation,
            )
            print(f'use_local_branch: True (GatedLocalFusion injected, branch_type={local_branch_type})')
        else:
            self.local_fusion = None

    def forward(self,x,text):
        # import pdb;pdb.set_trace()
        txt_feat = self.encode_text(text)
        feat = self.resnet50(x)
        global_visual = self.attnpool(feat)
        global_txt = self.txt_attnpool(feat,txt_feat)
        global_feat = torch.cat([global_visual,global_txt],dim=-1)
        base_output = self.head(global_feat)
        if self.local_fusion is not None:
            return self.local_fusion(base_output, feat)
        return base_output

    def encode_text(self,text):
        text = clip.tokenize(text,context_length=77,truncate=True).cuda()
        x = self.wte(text).type(self.dtype)
        x = x + self.txt_pos.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_map = self.txt_model(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        # 分组学习率（调整2）：
        #   Group A  IP-IQA backbone / original head      lr_scale = lr_scale (1x)
        #   Group B  E1-inherited local (proj/score/weight) lr_scale = 3x
        #   Group C  new random-init MSDA blocks            lr_scale = 10x
        #   Group D  gates (refine + outer)                 lr_scale = 10x
        #   original head 额外按 self.head_scale 放大（若开启）
        base_wd, base_nowd = [], []
        local_b_wd, local_b_nowd = [], []
        msda_wd, msda_nowd = [], []
        gate_params = []
        p_head, p_head_non_wd = [], []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            is_head = self.head_scale and n.startswith("head")
            is_nowd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
            if is_head:
                (p_head_non_wd if is_nowd else p_head).append(p)
            elif "refine_gate_logit" in n or "local_gate_logit" in n:
                gate_params.append(p)
            elif any(k in n for k in (
                "fine_channel", "coarse_channel",
                "fine_spatial", "coarse_spatial",
                ".cross.", ".fuse.",
            )):
                (msda_nowd if is_nowd else msda_wd).append(p)
            elif "local_fusion" in n and (
                "proj" in n or "score_head" in n or "weight_head" in n
            ):
                (local_b_nowd if is_nowd else local_b_wd).append(p)
            else:
                (base_nowd if is_nowd else base_wd).append(p)

        optim_params = [
            {"params": base_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": base_nowd, "weight_decay": 0, "lr_scale": lr_scale},
            {"params": local_b_wd, "weight_decay": weight_decay, "lr_scale": 3.0 * lr_scale},
            {"params": local_b_nowd, "weight_decay": 0, "lr_scale": 3.0 * lr_scale},
            {"params": msda_wd, "weight_decay": weight_decay, "lr_scale": 10.0 * lr_scale},
            {"params": msda_nowd, "weight_decay": 0, "lr_scale": 10.0 * lr_scale},
            {"params": gate_params, "weight_decay": 0, "lr_scale": 10.0 * lr_scale},
        ]
        if self.head_scale:
            optim_params.append(
                {"params": p_head, "weight_decay": weight_decay, "lr_scale": self.head_scale}
            )
            optim_params.append(
                {"params": p_head_non_wd, "weight_decay": 0, "lr_scale": self.head_scale}
            )
            print(f"head scale: {self.head_scale}")
        # 过滤空参数组，避免 AdamW 报错
        optim_params = [g for g in optim_params if len(g["params"]) > 0]
        return optim_params

    @classmethod
    def from_config(cls, cfg):
        base_ckpt = cfg.get('base_ckpt','../data/ckpt/clip/openai/resnet/RN50.pt')
        input_resolution = cfg.get("input_resolution",512)
        output_dim = cfg.get("output_dim",None)
        freeze_text = cfg.get("freeze_text",True)
        qa_token = cfg.get("qa_token",False)
        head_scale = cfg.get('head_scale',None)
        use_mlp_head = cfg.get('use_mlp_head',False)
        dropout_rate = cfg.get('dropout_rate',0.)
        use_local_branch = cfg.get('use_local_branch',False)
        local_hidden_dim = cfg.get('local_hidden_dim',256)
        local_gate_init = cfg.get('local_gate_init',-2.0)
        local_use_attention = cfg.get('local_use_attention',True)
        local_branch_type = cfg.get('local_branch_type','weighted')
        ms_num_heads = cfg.get('ms_num_heads',4)
        ms_mlp_ratio = cfg.get('ms_mlp_ratio',2.0)
        ms_refine_gate_init = cfg.get('ms_refine_gate_init',-2.0)
        ms_use_dual_attention = cfg.get('ms_use_dual_attention',True)
        ms_aggregation = cfg.get('ms_aggregation','weighted')

        model = cls(
                base_ckpt=base_ckpt,
                input_resolution=input_resolution,
                output_dim=output_dim,
                use_mlp_head=use_mlp_head,
                dropout_rate=dropout_rate,
                freeze_text=freeze_text,
                head_scale=head_scale,
                qa_token=qa_token,
                use_local_branch=use_local_branch,
                local_hidden_dim=local_hidden_dim,
                local_gate_init=local_gate_init,
                local_use_attention=local_use_attention,
                local_branch_type=local_branch_type,
                ms_num_heads=ms_num_heads,
                ms_mlp_ratio=ms_mlp_ratio,
                ms_refine_gate_init=ms_refine_gate_init,
                ms_use_dual_attention=ms_use_dual_attention,
                ms_aggregation=ms_aggregation,
            )

        load_finetuned = cfg.get("load_finetuned",False)  # you've loaded the clip weight in `__init__` func
        if load_finetuned:
            model.load_checkpoint_from_config(cfg)

        # 调整2：从已验证的 E1 checkpoint warm start（只加载 backbone/head/local 预测头，
        # 新增的 MSDA refinement 保持随机初始化；手动映射 E1 proj -> refiner.proj）
        warm_start_e1 = cfg.get("warm_start_e1", False)
        warm_start_ckpt = cfg.get("warm_start_ckpt", None)
        if warm_start_e1:
            assert warm_start_ckpt and os.path.exists(warm_start_ckpt), (
                f"warm_start_ckpt not found: {warm_start_ckpt}"
            )
            ckpt = torch.load(warm_start_ckpt, map_location="cpu")
            sd = ckpt["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)

            # 手动映射 E1 的 local_branch.proj -> E3 的 refiner.proj（结构同为 Sequential(Conv2d, GELU)）
            e1_proj_keys = [
                k for k in sd if k.startswith("local_fusion.local_branch.proj.")
            ]
            if e1_proj_keys:
                proj_sd = {
                    k.split("local_branch.proj.", 1)[1]: sd[k] for k in e1_proj_keys
                }
                branch = model.local_fusion.local_branch
                if hasattr(branch, "refiner"):
                    branch.refiner.proj.load_state_dict(proj_sd)
                    print("[warm_start_e1] copied E1 proj -> refiner.proj")
                else:
                    print("[warm_start_e1] WARNING: branch has no refiner, proj NOT copied")

            print(f"[warm_start_e1] loaded from {warm_start_ckpt}")
            print(f"[warm_start_e1] missing keys ({len(missing)}): {missing[:6]}")
            print(f"[warm_start_e1] unexpected keys ({len(unexpected)}): {unexpected[:6]}")

        return model