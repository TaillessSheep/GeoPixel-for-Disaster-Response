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
            has_img =  'image' in d[0].keys()
            has_mask = ('polygons' in d[0].keys()) or ('segmentation' in d[0].keys())

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
                assert isinstance(image_file, str), "image_file must be a string" #need single image
                image_g = Image.open(image_file).convert("RGB")
                w, h = image_g.size
                ori_hw = (h, w)
                image_g = self.vis_processor_gr(image_g)
                if 'polygons' in self.raw_data[i]:
                    polygons_file = self.raw_data[i]['polygons']
                    assert isinstance(polygons_file, str), "polygons_file must be a string"
                    with open(polygons_file, 'r') as file:
                        try:
                            data = json.load(file)  
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON file: {polygons_file}")
                    # Processing the polygons data
                    masks = []
                    for polygon in data["polygons"]:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for poly in polygon:
                            assert len(poly) > 0 and len(poly[0]) == 2, "invalid multiple polygons"
                            cv2.fillPoly(mask, np.array([poly], dtype=np.int32), color=1)
                        masks.append(mask)
                    assert len(masks) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image_file}"

                elif 'segmentation' in self.raw_data[i]:

                    segm = self.raw_data[i]['segmentation']
                    assert len(segm) == conv_text.count('[SEG]') , f"number of grounding tokens are not equal to number of masks provided with image: {image_file}"
                    masks = []
                    if segm is None:
                        raise ValueError(f"Failed to read mask")
                    for rle in segm:
                        binary_mask = M.decode(rle).astype(np.uint8)
                        masks.append(binary_mask)
                else:
                    print(f"No 'polygon' or 'segmentation' found in grounding data")
                
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
