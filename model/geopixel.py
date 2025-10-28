from typing import List, Optional, Tuple, Union

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.IXC.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from model.IXC.modeling_internlm2 import InternLM2Model
from model.sam2.build_sam import build_sam2_hf
from model.sam2.utils.transforms import SAM2Transforms
from transformers import TextStreamer
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GeoPixelMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelMetaModel, self).__init__(config)
        self.config = config
        self.config.train_mask_decoder = getattr(self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False))
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 256))
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_geopixel_modules(self.config)

    def initialize_geopixel_modules(self, config):
        # grounding vision model
        self.visual_model = build_sam2_hf(self.vision_pretrained)

        self._transform = SAM2Transforms(
                    resolution=self.visual_model.image_size,
                    mask_threshold=0.0,
                    max_hole_area=0.0,
                    max_sprinkle_area=0.0,
                )
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # text projection layer
        in_dim = config.hidden_size 
        out_dim = config.out_dim    
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GeoPixelModel(GeoPixelMetaModel, InternLM2Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GeoPixelModel, self).__init__(config, **kwargs)
        self.config.use_cache = False


class GeoPixelForCausalLM(InternLMXComposer2ForCausalLM):
    def __init__(self,config,**kwargs,):
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)
        self.model = GeoPixelModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def encode_g_img(self, image):
        """
        Calculates the image embeddings for the provided image
        Arguments:
          image (np.ndarray or str)
        """
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp','.tif'}:
                image = Image.open(image)
                w, h = image.size
                _orig_hw = [(h, w)] 
            else:
                print ('Unknow input format', image)
                return None
        else:
            assert isinstance(image, torch.Tensor)
            _orig_hw = [image.shape[:2]]
        image = self.model._transform(image)
        image = image[None, ...].to(self.device)
        assert ( len(image.shape) == 4 and image.shape[1] == 3), f"image must be of size 1x3xHxW, got {image.shape}"
        features = self.get_visual_embs(image)   
        return features,_orig_hw

    def get_visual_embs(self, img_batch: torch.FloatTensor):
        with torch.no_grad():
            torch.cuda.empty_cache()
            img_batch = img_batch.to(self.device)
            batch_size = img_batch.shape[0]
            assert (
                len(img_batch.shape) == 4 and img_batch.shape[1] == 3
            ), f"grounding_img_batch must be of size Bx3xHxW, got {img_batch.shape}"
            backbone_out = self.model.visual_model.forward_image(img_batch)
            _, vision_feats, _, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
            if self.model.visual_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.model._bb_feat_sizes[::-1])
            ][::-1]
            features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features
    
    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)
    
    def model_forward(
            self,
            inference: bool = False,
            **kwargs,
    ):
        samples = kwargs.get('samples', None)
        if samples and samples['data_type'][0] == 'grounding': 
            kwargs['output_hidden_states'] = True
            kwargs['use_cache'] = False

            torch.cuda.empty_cache()
            outputs = super().forward(**kwargs)

            if inference:
                assert len(samples['text_input']) == 1 and len(samples['image'][0]) == 1 #single image and single query
                output_hidden_states = [outputs.hidden_states]
                outputs = None
            else:
                output_hidden_states = outputs.hidden_states

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            seg_token_mask = outputs.seg_token_mask
            pred_embeddings = [states[masks] for states, masks in zip(last_hidden_state, seg_token_mask)]
            image_g_batch = torch.cat(samples['image_g'][0],dim = 0)
            image_g_features = self.get_visual_embs(image_g_batch)
            ori_hw = samples['ori_hw'][0]
            all_pred_masks = []
            for i in range(len(pred_embeddings)): #(bs,)
                if (pred_embeddings[i].numel()== 0):
                    pred_masks.append([])
                    continue
                (sparse_embeddings, dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                batch_mode = (pred_embeddings[i].shape[0]>1)
                high_res_features = [
                    feat_level[i].unsqueeze(0)
                    for feat_level in image_g_features["high_res_feats"]
                ]
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                image_g_embeds = image_g_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)
                low_res_masks, _, _ , _ = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=image_g_embeds,
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    repeat_image=batch_mode,
                    multimask_output=False,
                    high_res_features=high_res_features,
                )
                pred_masks = self.model._transform.postprocess_masks(
                    low_res_masks,
                    ori_hw[i],
                )
                all_pred_masks.append(pred_masks[:, 0])
                

            model_output = outputs
            gt_masks =  samples['masks'][0]
            pred_masks = all_pred_masks 

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0

            for batch_idx in range(len(pred_masks)): # for every image
                cur_gt_masks = torch.stack(
                    [
                        torch.from_numpy(gt_mask).to(dtype=pred_masks[batch_idx].dtype, device=pred_masks[batch_idx].device)
                        for gt_mask in gt_masks[batch_idx]
                    ],
                    dim=0
                ) # expected (bs,H,W)
                cur_pred_masks = pred_masks[batch_idx]
                assert (
                    cur_gt_masks.shape[0] == cur_pred_masks.shape[0]
                ), "gt_masks.shape: {}, pred_masks.shape: {}".format(
                    cur_gt_masks.shape, cur_pred_masks.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(cur_pred_masks, cur_gt_masks, num_masks=cur_gt_masks.shape[0])
                    * cur_gt_masks.shape[0]
                )
                num_masks += cur_gt_masks.shape[0] 

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = ce_loss + mask_loss
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=model_output.logits,
                past_key_values=model_output.past_key_values,
                hidden_states=output_hidden_states,
                attentions=model_output.attentions,
            )
            outputs.ce_loss = ce_loss
            outputs.mask_bce_loss = mask_bce_loss
            outputs.mask_dice_loss = mask_dice_loss
            outputs.mask_loss = mask_loss
        else: 
            outputs =  super().forward(**kwargs)
        return outputs

    def evaluate(
        self,
        tokenizer,
        query: str,
        images: List[Tuple[str, str]] = [],
        hd_num: int = 9,
        history: List[Tuple[str, str]] = [],
        max_new_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ):
        with torch.no_grad():
            inputs, im_mask, _ = self.interleav_wrap_chat(query, images, history=history, hd_num=hd_num)        
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items() if torch.is_tensor(v)
            }
            eos_token_id = [
                tokenizer.eos_token_id,
                #tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
            ]
            all_pred_masks = []
            
            if stream:
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            else: 
                streamer = None

            outputs = self.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                im_mask=im_mask,
                input_ids = None,
                streamer= streamer,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p= 1.0,
                top_k = 0,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,
                infer_mode = 'base',
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            output_ids = outputs['sequences']
            response = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
            response = response.replace("[UNUSED_TOKEN_145]","")
            history = history + [(query, response)]
            if len(images)==1 and isinstance(images[0], str): 
                output_hidden_states = outputs.hidden_states[-1] 
                seg_token_mask = output_ids[:, 1:-1] == self.seg_token_idx
                inputs_embeds_len = inputs['inputs_embeds'].size(1)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], inputs_embeds_len)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )
                hidden_states = []
                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = [states[masks] for states, masks in zip(last_hidden_state, seg_token_mask)]
                image_g_features, ori_hw = self.encode_g_img(images[0]) 

                for i in range(len(pred_embeddings)):
                    if (pred_embeddings[i].numel()== 0):
                        all_pred_masks.append([])
                        continue
                    (sparse_embeddings,dense_embeddings,) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )
                    batch_mode = (pred_embeddings[i].shape[0]>1)
                    high_res_features = [
                        feat_level[i].unsqueeze(0)
                        for feat_level in image_g_features["high_res_feats"]
                    ]
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    image_g_embeds = image_g_features['image_embed'][i].unsqueeze(0).to(torch.bfloat16)

                    low_res_masks, _, _ , _  = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=image_g_embeds,
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        repeat_image=batch_mode,
                        multimask_output=False,
                        high_res_features=high_res_features,
                    )
                    pred_masks = self.model._transform.postprocess_masks(
                        low_res_masks,
                        ori_hw[i],
                    )
                    all_pred_masks.append(pred_masks[:, 0])

        return response, all_pred_masks