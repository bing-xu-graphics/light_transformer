import torch.nn as nn
import torch
from .utils import index, dist_mat
from pointnet2_ops import pointnet2_utils
from .pointtransformerv3 import PointTransformerV3


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(3, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


class Model(nn.Module):
    def __init__(self, DATA_CONFIG, depth, dim, k=8):
        super(Model, self).__init__()
        self.DATA_CONFIG = DATA_CONFIG

        self.pre_fc_scene = nn.Sequential(
            nn.Conv1d(12, dim//4, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//4),
            nn.ReLU(),
            nn.Conv1d(dim//4, dim//4, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//4),
            nn.ReLU(),
        )
        self.group1_fc_scene = nn.Sequential(
            nn.Conv1d(dim//4, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
            nn.Conv1d(dim//2, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(),
        )
        self.group2_fc_scene = nn.Sequential(
            nn.Conv1d(dim//2, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.enc = PointTransformerV3(
            in_channels=dim,
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(dim, dim, dim, dim * 2, dim * 2),
            enc_num_head=(dim // 32, dim // 32, dim // 32, dim // 16, dim // 16),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(dim, dim, dim * 2, dim * 2),
            dec_num_head=(dim // 32, dim // 32, dim // 16, dim // 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            pdnorm_decouple=False,
            pdnorm_affine=False,
        )
        
        self.pre_fc_query = nn.Sequential(
            nn.Conv1d(9, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.dec_blocks = nn.ModuleList([])
        for i in range(depth):
            trans = Transformer2D(
                dim=dim, 
                ff_hid_dim=int(4*dim),
                ff_dp_rate=0.1, 
                attn_dp_rate=0.1,
            )
            self.dec_blocks.append(trans)
        self.out_fc = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Softplus(),
        )
        self.k = k
    
    @staticmethod
    def knn(pc1, pc2, feat1, feat2, k=4):
        dist = dist_mat(pc1, pc2)
        knn_indx = torch.topk(dist, dim=2, k=k, largest=False, sorted=True)[1]
        knn_feat2 = index(feat2, knn_indx)

        if feat1 is not None:
            knn_feat2_norm = knn_feat2 - feat1[:, :, None, :] 
            return torch.cat((knn_feat2_norm, feat1[:, :, None, :]), dim=-2)
        else:
            knn_pc = index(pc2, knn_indx)
            knn_pc_norm = knn_pc - pc1[:, :, None, :]
            return knn_feat2, knn_pc_norm

    @staticmethod
    def fps(pc, n_pts, feat, return_pc=False):
        fps_indx = pointnet2_utils.furthest_point_sample(pc.contiguous(), n_pts).long()
        feat_fps = index(feat, fps_indx)
        if return_pc:
            pc_fps = index(pc, fps_indx)
            return feat_fps, pc_fps
        return feat_fps


    def forward(self, scene_data, query_data):
        # extract scene data
        SCENE_OFFSET_POS = self.DATA_CONFIG["scene_data"]["OFFSET_POS"]
        SCENE_OFFSET_NORMAL = self.DATA_CONFIG["scene_data"]["OFFSET_NORMAL"]
        SCENE_OFFSET_ALBEDO = self.DATA_CONFIG["scene_data"]["OFFSET_ALBEDO"]
        SCENE_OFFSET_EMISSIVENESS = self.DATA_CONFIG["scene_data"]["OFFSET_EMISSIVENESS"]

        scene_pos = scene_data[..., SCENE_OFFSET_POS:SCENE_OFFSET_POS + 3]  # (batch_size, num_scene_points, 3)

        scene_data = scene_data.permute(0, 2, 1)
        scene_emb = self.pre_fc_scene(scene_data)
        scene_emb = scene_emb.permute(0, 2, 1)
        
        scene_emb_fps, scene_pos_fps = Model.fps(scene_pos, scene_pos.shape[1] // 2, scene_emb, return_pc=True)
        scene_emb_knn = Model.knn(scene_pos_fps, scene_pos, scene_emb_fps, scene_emb, k=32)
        B, N, K, _ = scene_emb_knn.shape
        scene_emb_knn = scene_emb_knn.reshape(B*N, K, -1).permute(0, 2, 1)
        scene_emb = self.group1_fc_scene(scene_emb_knn).max(dim=-1)[0]
        scene_emb = scene_emb.reshape(B, N, -1)

        scene_pos = scene_pos_fps
        scene_emb_fps, scene_pos_fps = Model.fps(scene_pos, scene_pos.shape[1] // 2, scene_emb, return_pc=True)
        scene_emb_knn = Model.knn(scene_pos_fps, scene_pos, scene_emb_fps, scene_emb, k=32)
        B, N, K, _ = scene_emb_knn.shape
        scene_emb_knn = scene_emb_knn.reshape(B*N, K, -1).permute(0, 2, 1)
        scene_emb = self.group2_fc_scene(scene_emb_knn).max(dim=-1)[0]
        scene_emb = scene_emb.reshape(B, N, -1)

        scene_pos = scene_pos_fps
        extra_loss = {}

        inp = {
            "coord": scene_pos.reshape(-1, scene_pos.shape[-1]), 
            "grid_size": 0.01, 
            "feat": scene_emb.reshape(-1, scene_emb.shape[-1]),
            "offset": torch.tensor([int((i + 1) * scene_pos.shape[1]) for i in range(scene_pos.shape[0])]).to(scene_data.device)
        }
        scene_emb = self.enc(inp)
        scene_emb = scene_emb.feat.reshape(scene_pos.shape[0], scene_pos.shape[1], -1)
        
        QUERY_OFFSET_POS = self.DATA_CONFIG["query_data"]["OFFSET_POS"]
        QUERY_OFFSET_NORMAL = self.DATA_CONFIG["query_data"]["OFFSET_NORMAL"]
        QUERY_OFFSET_ALBEDO = self.DATA_CONFIG["query_data"]["OFFSET_ALBEDO"]
        
        query_pos = query_data[..., QUERY_OFFSET_POS:QUERY_OFFSET_POS + 3]  # (batch_size, num_query_points, 3)

        query_data = query_data.permute(0, 2, 1)
        query_emb = self.pre_fc_query(query_data)
        query_emb = query_emb.permute(0, 2, 1)
        ret_latents, pos = Model.knn(query_pos, scene_pos, feat1=None, feat2=scene_emb, k=self.k)
        for i in range(len(self.dec_blocks)):
            query_emb = self.dec_blocks[i](query_emb, ret_latents, pos)

        out = self.out_fc(query_emb)
        return out, extra_loss

    def encode_scene(self, scene_data):
        scene_pos = scene_data[..., :3]
        scene_emb = self.pre_fc_scene(scene_data.permute(0, 2, 1)).permute(0, 2, 1)

        scene_emb_fps, scene_pos_fps = Model.fps(
            scene_pos, scene_pos.shape[1] // 2, scene_emb, return_pc=True
        )
        scene_emb_knn = Model.knn(
            scene_pos_fps, scene_pos, scene_emb_fps, scene_emb, k=32
        )
        batch, points, neighbors, _ = scene_emb_knn.shape
        scene_emb_knn = scene_emb_knn.reshape(
            batch * points, neighbors, -1
        ).permute(0, 2, 1)
        scene_emb = self.group1_fc_scene(scene_emb_knn).max(dim=-1)[0]
        scene_emb = scene_emb.reshape(batch, points, -1)

        scene_pos = scene_pos_fps
        scene_emb_fps, scene_pos_fps = Model.fps(
            scene_pos, scene_pos.shape[1] // 2, scene_emb, return_pc=True
        )
        scene_emb_knn = Model.knn(
            scene_pos_fps, scene_pos, scene_emb_fps, scene_emb, k=32
        )
        batch, points, neighbors, _ = scene_emb_knn.shape
        scene_emb_knn = scene_emb_knn.reshape(
            batch * points, neighbors, -1
        ).permute(0, 2, 1)
        scene_emb = self.group2_fc_scene(scene_emb_knn).max(dim=-1)[0]
        scene_emb = scene_emb.reshape(batch, points, -1)
        scene_pos = scene_pos_fps

        inp = {
            "coord": scene_pos.reshape(-1, 3),
            "grid_size": 0.01,
            "feat": scene_emb.reshape(-1, scene_emb.shape[-1]),
            "offset": torch.tensor(
                [
                    int((i + 1) * scene_pos.shape[1])
                    for i in range(scene_pos.shape[0])
                ],
                device=scene_data.device,
            ),
        }
        scene_emb = self.enc(inp).feat.reshape(
            scene_pos.shape[0], scene_pos.shape[1], -1
        )

        return scene_emb, scene_pos

    def decode_queries(self, scene_emb, scene_pos, query_data):
        query_pos = query_data[..., :3]
        query_emb = self.pre_fc_query(query_data.permute(0, 2, 1)).permute(0, 2, 1)
        local_latents, relative_pos = Model.knn(
            query_pos, scene_pos, feat1=None, feat2=scene_emb, k=self.k
        )
        for block in self.dec_blocks:
            query_emb = block(query_emb, local_latents, relative_pos)
        return self.out_fc(query_emb)
