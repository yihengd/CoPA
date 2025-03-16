import torch
from torch import nn
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from .utils import FFN


class CoPA(nn.Module):
    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()

        self.concept_list = concept_list
        self.config = config
        self.model_init(model_name, config)
        self.concept_num = len(self.concept_list.keys())
        self.visual_anchors = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.concept_num, 768)))

        # embedding layer
        self.embed_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.embed_norm = nn.LayerNorm(768)
        self.embed_ffn = FFN(768, 768 * 4)
        self.embed_cpt()

        # transformer block
        layers = [layer for layer in self.model.visual.trunk.blocks]
        num_layers = len(self.model.visual.trunk.blocks)
        self.layer_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True) for _ in range(num_layers)
        ])
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(768) for _ in range(num_layers)
        ])
        self.layer_ffn = nn.ModuleList([
            FFN(768, 768 * 4) for _ in range(num_layers)
        ])
        self.layer_count = 0
        self.concept_embed_list = []
        self.block_cpt(layers)

        self.embed_agg = nn.Linear(in_features=num_layers, out_features=1)
        self.proj = nn.Linear(in_features=768, out_features=512, bias=False)
        self.gate = nn.Linear(in_features=self.concept_num, out_features=1, bias=False)
        self.cls_head = nn.Linear(in_features=512, out_features=config.num_class)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, imgs):
        self.concept_embed_list.clear()
        self.layer_count = 0
        _ = self.model(imgs, None)
        B = imgs.shape[0]

        concept_embeds = torch.stack(self.concept_embed_list, dim=-1)
        multilayer_embed = self.embed_agg(concept_embeds).squeeze(-1)
        multilayer_embed = F.normalize(self.proj(multilayer_embed), dim=-1)

        image_logits_dict = {}
        concept_feat_list = []
        for idx, key in enumerate(self.concept_token_dict.keys()):
            text_feature = self.concept_token_dict[key].repeat(B, 1, 1)
            visual_feature = self.logit_scale * multilayer_embed[:, idx:idx + 1, :]
            logit = (visual_feature @ text_feature.permute(0, 2, 1)).squeeze(1)
            logit = F.softmax(logit, dim=-1)
            image_logits_dict[key] = logit

            logit_temp = logit.unsqueeze(1)
            concept_feat = (logit_temp @ text_feature).squeeze(1)
            concept_feat_list.append(concept_feat)

        concept_agg_feature = torch.stack(concept_feat_list, dim=-1)
        diagnose_feature = self.gate(concept_agg_feature).squeeze(-1)
        cls_logits = self.cls_head(diagnose_feature)
        cls_logits = torch.softmax(cls_logits, dim=1)

        return cls_logits, image_logits_dict

    def model_init(self, model_name, config):
        self.model_name = model_name
        if self.model_name == 'biomedclip':
            self.model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        elif self.model_name == 'openclip':
            self.model, _ = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        self.model.cuda()

        concept_keys = list(self.concept_list.keys())
        self.concept_token_dict = {}
        for key in concept_keys:
            prefix = f"this is a dermoscopic image, the {key} of the lesion is "
            attr_concept_list = self.concept_list[key]
            prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
            tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
            _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
            self.concept_token_dict[key] = tmp_concept_feats.detach()

        self.logit_scale = logit_scale.detach()

    def embed_cpt(self):
        def embed_prompt_concat(module, input, output):
            feat_map = output[:, 1:, :].detach()
            B, _, _ = feat_map.shape
            query = self.visual_anchors.repeat(B, 1, 1)

            concept_embed, _ = self.embed_attn(query, feat_map, feat_map)
            concept_embed = self.embed_norm(self.embed_ffn(concept_embed) + query)
            output = torch.cat([
                output[:, :1, :],
                concept_embed,
                output[:, 1:, :]
            ], dim=1)
            return output

        self.model.visual.trunk.norm_pre.register_forward_hook(embed_prompt_concat)

    def block_cpt(self, layers):
        def prompt_concat(module, input, output):
            attn = self.layer_attn[self.layer_count]
            norm = self.layer_norm[self.layer_count]
            ffn = self.layer_ffn[self.layer_count]

            feat_map = output[:, (1 + self.concept_num):, :].detach()
            B, _, _ = feat_map.shape
            query = self.visual_anchors.repeat(B, 1, 1)

            concept_embed, _ = attn(query, feat_map, feat_map)
            concept_embed = norm(ffn(concept_embed) + query)
            self.concept_embed_list.append(concept_embed)
            output = torch.cat([
                output[:, :1, :],
                concept_embed,
                output[:, (1 + self.concept_num):, :]
            ], dim=1)
            self.layer_count += 1
            return output

        for layer in layers:
            layer.register_forward_hook(prompt_concat)
