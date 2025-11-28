import random
import cv2
import json
import numpy as np
from ixc_utils import R560_HD18_Identity_transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from model.sam2.utils.transforms import SAM2Transforms
from pycocotools import mask as M

def conv2text(sources):
    END_HUMAN = '[UNUSED_TOKEN_145]\n'
    END_BOT = '[UNUSED_TOKEN_145]\n'
    conversation = ''

    for idx, sentence in enumerate(sources):
        BEGIN_SIGNAL = ''

        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':
            from_str = '[UNUSED_TOKEN_146]user\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() +
                END_HUMAN)
        else:
            from_str = '[UNUSED_TOKEN_146]assistant\n'
            temp = (
                BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_BOT)
        conversation += temp

    return conversation + '</s>'


class ImageProcessorHD:

    def __init__(self, resolution=560, hd_num=18):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.resolution = resolution
        self.hd_num = hd_num
        print(f'hd_num = {self.hd_num}')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        item = Image.open(item).convert('RGB')
        return self.transform(
            R560_HD18_Identity_transform(
                item, resolution=self.resolution, hd_num=self.hd_num))


class Mix_dataset(Dataset):

    def __init__(self,
                json_datas,
                batch_size=1,
                local_rank=0,
                resolution=560,
                resolution_gr = 1024,
                hd_num=18):
        """vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file."""
        super().__init__()
        print(f'initializing mix data at rank {local_rank}')
        self.datasets_text, self.datasets_multi, self.datasets_grounding = [], [], []
        self.data_num_text, self.data_num_multi, self.data_num_grounding = [], [], []

        self.batch_size = batch_size
        self.set_seed = False
        self.local_rank = local_rank
        for _, d in json_datas.items():
            has_img = 'image' in d[0].keys()
            has_mask = ('polygons' in d[0].keys()) or ('polygon' in d[0].keys()) or ('segmentation' in d[0].keys())

            sub_data_set = Sample_dataset(
                d, 
                batch_size, 
                has_img=has_img,
                has_mask=has_mask,
                resolution=resolution, 
                resolution_gr=resolution_gr, 
                hd_num=hd_num
            ) 
            if has_img:
                if has_mask:
                    self.datasets_grounding.append(sub_data_set)
                    self.data_num_grounding.append(len(sub_data_set))
                else:
                    self.datasets_multi.append(sub_data_set)
                    self.data_num_multi.append(len(sub_data_set))
            else:
                self.datasets_text.append(sub_data_set)
                self.data_num_text.append(len(sub_data_set))
        
        self.data_ratio_grounding = [
            float(ratio) / sum(self.data_num_grounding)
            for ratio in self.data_num_grounding
        ]
        self.data_ratio_multi = [
            float(ratio) / sum(self.data_num_multi)
            for ratio in self.data_num_multi
        ]
        self.data_ratio_text = [
            float(ratio) / sum(self.data_num_text)
            for ratio in self.data_num_text
        ]
        self.data_num = np.sum(self.data_num_grounding) + np.sum(self.data_num_multi) + np.sum(self.data_num_text)
        self.num_of_ds =sum(1 for dataset in [self.datasets_text, self.datasets_multi, self.datasets_grounding] if dataset)
        self.use_grounding = 0
        self.use_multi = batch_size*(self.num_of_ds-1)  #equal mixing

    def __len__(self):
        return int(self.data_num / self.batch_size)

    def __getitem__(self, index):
        if not self.set_seed:
            random.seed(index)
            self.set_seed = True
            print(f'Set seed {index} for rank {self.local_rank}')

        if len(self.datasets_grounding) == 0 and len(self.datasets_multi) == 0 and len(self.datasets_text) == 0:
            raise ValueError(
                'All _grounding, _multi and _text are empty. Cannot sample any data.')
        
        if len(self.datasets_grounding) > 0 and (self.use_grounding < self.batch_size
                                             or ( len(self.datasets_multi) == 0 and len(self.datasets_text) == 0 )):
            data_idx = random.choices(
                range(len(self.data_ratio_grounding)),
                weights=self.data_ratio_grounding,
                k=1)[0]
            sample = self.datasets_grounding[data_idx].get_item()
        elif len(self.datasets_multi) > 0 and (self.use_multi < self.batch_size
                                             or len(self.datasets_text) == 0):
            data_idx = random.choices(
                range(len(self.data_ratio_multi)),
                weights=self.data_ratio_multi,
                k=1)[0]
            sample = self.datasets_multi[data_idx].get_item()
        elif len(self.datasets_text) > 0:
            data_idx = random.choices(
                range(len(self.data_ratio_text)),
                weights=self.data_ratio_text,
                k=1)[0]
            sample = self.datasets_text[data_idx].get_item()
        else:
            raise ValueError('Unable to select a dataset for sampling.')
        
        self.use_grounding += 1
        self.use_multi += 1
        if self.use_grounding == self.batch_size * self.num_of_ds:
            self.use_grounding = 0
        if self.use_multi == self.batch_size * self.num_of_ds:
            self.use_multi = 0
        return dict(samples=sample)


class Sample_dataset(Dataset):

    def __init__(self,
                 raw_data,
                 batch_size,
                 has_img=False,
                 has_mask=False,
                 resolution=560,
                 resolution_gr = 1024,
                 hd_num=18):
        self.raw_data = raw_data
        print(f'initilized Sample_dataset with {len(self.raw_data)}')
        self.batch_size = batch_size
        self.vis_processor = ImageProcessorHD(
            resolution=resolution, hd_num=hd_num)
        self.vis_processor_gr = SAM2Transforms(
            resolution=resolution_gr,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0)
        self.text_processor = conv2text
        self.has_img = has_img
        self.has_mask = has_mask

    def __len__(self):
        return len(self.raw_data)

    def __get_item__(self, i):
        conv_text = conv2text(self.raw_data[i]['conversations'])
        seg_count = conv_text.count('[SEG]')

        sample = dict(text_input=conv_text, )
        if self.has_img:
            image_file = self.raw_data[i]['image']
            if type(image_file) == str:
                image = self.vis_processor(image_file)
            elif type(image_file) == list:
                image = [self.vis_processor(i) for i in image_file]
            else:
                raise NotImplementedError('Image format not supported')
            sample['image'] = image
            if self.has_mask:
                assert isinstance(image_file, str), "image_file must be a string"  # need single image
                image_g = Image.open(image_file).convert("RGB")
                w, h = image_g.size
                ori_hw = (h, w)
                image_g = self.vis_processor_gr(image_g)

                # 修改开始：支持多种格式的掩码数据
                masks = []

                if 'polygons' in self.raw_data[i]:
                    polygons_file = self.raw_data[i]['polygons']

                    # 情况1：polygons是字符串（文件路径）
                    if isinstance(polygons_file, str):
                        with open(polygons_file, 'r') as file:
                            try:
                                data = json.load(file)
                            except json.JSONDecodeError:
                                raise ValueError(f"Invalid JSON file: {polygons_file}")

                        # 从文件中读取多边形数据
                        polygon_data = data.get("polygons", data)  # 兼容两种格式
                    # 情况2：polygons是直接内嵌的数据（列表）
                    elif isinstance(polygons_file, list):
                        polygon_data = polygons_file
                    else:
                        raise ValueError(f"polygons must be string or list, got {type(polygons_file)}")

                    # 处理多边形数据
                    for polygon in polygon_data:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        # 多边形可能以不同格式存储
                        if len(polygon) > 0 and isinstance(polygon[0][0], (int, float)):  # 单层多边形
                            cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), color=1)
                        else:  # 多层多边形（包含多个轮廓）
                            for poly in polygon:
                                cv2.fillPoly(mask, np.array([poly], dtype=np.int32), color=1)
                        masks.append(mask)

                elif 'polygon' in self.raw_data[i]:  # 注意：有的数据可能是'polygon'单数形式
                    polygon_data = self.raw_data[i]['polygon']
                    if not isinstance(polygon_data, list):
                        raise ValueError(f"polygon must be list, got {type(polygon_data)}")

                    # 处理多边形数据
                    for polygon in polygon_data:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        if len(polygon) > 0 and isinstance(polygon[0][0], (int, float)):  # 单层多边形
                            cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), color=1)
                        else:  # 多层多边形
                            for poly in polygon:
                                cv2.fillPoly(mask, np.array([poly], dtype=np.int32), color=1)
                        masks.append(mask)

                elif 'segmentation' in self.raw_data[i]:
                    segm = self.raw_data[i]['segmentation']
                    if segm is None:
                        raise ValueError(f"Failed to read mask")
                    for rle in segm:
                        binary_mask = M.decode(rle).astype(np.uint8)
                        masks.append(binary_mask)

                # 修改第 189-194 行
                if seg_count == 1 and len(masks) > 1:
                    print(f"Auto-merging {len(masks)} masks into 1 semantic mask")
                    # 创建合并的语义掩码
                    merged_mask = np.zeros((h, w), dtype=np.uint8)
                    for mask in masks:
                        merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
                    # 替换为单个合并掩码
                    masks = [merged_mask]
                    # 同时更新标注信息，表明这是语义掩码
                    self.raw_data[i]['is_semantic_mask'] = True


                else:
                    print(f"No 'polygon' or 'segmentation' found in grounding data")
                    sample['image_g'] = image_g
                    sample['ori_hw'] = ori_hw
                    sample['masks'] = []

                    # === 新增：确保masks列表不为空 ===
                    if len(sample['masks']) == 0:
                        print(f"Warning: Empty masks list for item {i}, creating dummy mask")
                        # 创建最小化的dummy mask
                        h, w = ori_hw
                        dummy_mask = np.zeros((h, w), dtype=np.uint8)
                        # 在中心添加一个小方块（图像尺寸的2%）
                        center_h, center_w = h // 2, w // 2
                        size = max(1, min(h, w) // 50)
                        h1 = max(0, center_h - size)
                        h2 = min(h, center_h + size)
                        w1 = max(0, center_w - size)
                        w2 = min(w, center_w + size)
                        dummy_mask[h1:h2, w1:w2] = 1
                        sample['masks'] = [dummy_mask]
                    # === 新增结束 ===

                    return sample

                # 验证掩码数量与[SEG]标记数量一致
                seg_count = conv_text.count('[SEG]')
                if seg_count > 0 and len(masks) < seg_count:
                    # 允许一个[SEG]对应多个掩码
                    print(f"Warning: {len(masks)} masks for {seg_count} [SEG] tokens. "
                          f"This is allowed for semantic segmentation.")
                    # 如果完全没有掩码才报错
                    if len(masks) == 0:
                        raise ValueError(
                            f"Found {seg_count} [SEG] tokens but no masks provided "
                            f"with image: {image_file}"
                        )
                # 修改结束

                sample['image_g'] = image_g
                sample['ori_hw'] = ori_hw
                sample['masks'] = masks
            else: 
                sample['image_g'] = None
                sample['ori_hw'] = None
                sample['masks'] = None
        else:
            sample['image'] = None
        return sample

    def get_item(self, ):
        text_input, image, image_g, masks, ori_hw = [], [], [], [], []

        for i in range(self.batch_size):
            idx = random.randrange(len(self.raw_data))
            sample = self.__get_item__(idx)
            text_input.append(sample['text_input'])

            if sample['image'] is None:
                pass
            else:
                images_batch = []       # list of 1xCxHxW
                if type(sample['image']) is list:
                    for im in sample['image']:
                        images_batch.append(im.unsqueeze(0))
                else:
                    images_batch.append(sample['image'].unsqueeze(0))
                    if sample['image_g'] is None:
                        pass
                    else:
                        image_g.append(sample['image_g'].unsqueeze(0))
                        masks.append(sample['masks'])
                        ori_hw.append(sample['ori_hw'])
                image.append(images_batch)
        if self.has_mask:
            data_type = 'grounding' 
        elif self.has_img : 
            data_type = 'multi' 
        else:
            data_type = 'text'
        sample = {
            'text_input': text_input,
            'data_type': data_type,
        }
        if self.has_img:
            sample['image'] = image
        if self.has_mask:
            sample['image_g'] = image_g
            sample['ori_hw'] = ori_hw
            sample['masks'] = masks
        return sample
