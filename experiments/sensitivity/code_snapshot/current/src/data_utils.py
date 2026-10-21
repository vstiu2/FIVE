import json
import os
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

base_path = Path(__file__).absolute().parents[3].absolute()
data_path = base_path / 'data'

PIL.Image.MAX_IMAGE_PIXELS = 1024 * 1024 * 1024 // 3


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(data_path / 'fashion-iq' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(data_path / 'fashion-iq' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = data_path / 'fashion-iq' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = data_path / 'fashion-iq' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image['pixel_values'], target_image['pixel_values'], image_captions

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = data_path / 'fashion-iq' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = data_path / 'fashion-iq' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image['pixel_values']

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(data_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(data_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = data_path / 'cirr_dataset/images' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = data_path / 'cirr_dataset/images' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image['pixel_values'], target_image['pixel_values'], rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = data_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image['pixel_values']

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


def _custom_cirr_dataset_path() -> Path:
    packaged = Path(__file__).resolve().parents[1] / 'data' / 'CustomCIRR_graph'
    default = (packaged if packaged.exists()
               else Path('/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized'))
    return Path(os.environ.get('CUSTOM_CIRR_ROOT', default))


custom_cirr_dataset_path = _custom_cirr_dataset_path()


class CustomCIRRDataset(Dataset):
    """
    Custom CIR dataset compatible with the CIRR training/validation pipeline.
    Image absolute paths are used as image identifiers to keep classic and relative
    modes consistent.

    Modes and splits:
        - classic:  yields (image_path_str, image_tensor) for all images in the split
        - relative + train:  yields (ref_image_tensor, tgt_image_tensor, query_text)
        - relative + val:    yields (ref_path, tgt_path, query_text, candidate_paths)
        - relative + test:   yields (pair_id, ref_path, query_text, candidate_paths)
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        split_dir = custom_cirr_dataset_path / split
        with open(split_dir / f'{split}_query_text_with_graph.json') as f:
            self.triplets = json.load(f)

        with open(split_dir / f'{split}_split_image.json') as f:
            self.image_paths: List[str] = json.load(f)

        print(f"CustomCIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                entry = self.triplets[index]
                ref_path = entry['ref_image']
                rel_caption = entry['query_text']
                group_members = entry['candidate']

                if self.split == 'train':
                    ref_image = self.preprocess(PIL.Image.open(ref_path))
                    tgt_image = self.preprocess(PIL.Image.open(entry['tgt_image']))
                    return ref_image['pixel_values'], tgt_image['pixel_values'], rel_caption

                elif self.split == 'val':
                    return ref_path, entry['tgt_image'], rel_caption, group_members

                elif self.split == 'test':
                    return ref_path, entry['tgt_image'], rel_caption, group_members

            elif self.mode == 'classic':
                image_path = self.image_paths[index]
                image = self.preprocess(PIL.Image.open(image_path))
                return image_path, image['pixel_values']

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_paths)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


patternnet_dataset_path = Path('/data/lj/cir_images/PatternNet')
patterncir_graph_path = Path('/data/lj/CIR/datasets/patterncir_with_graph_normalized')


class PatternCIRDataset(Dataset):
    """
    PatternNet CIR dataset. Caption and image_splits schemas match CIRR; the
    only difference is that name->path entries are split-scoped (relative to
    {root}/{split}/) instead of sharing a single image root. The dataset can be
    used in 'relative' or 'classic' mode and yields the same tuples as
    CIRRDataset.

    When with_graph=True, self.triplets is loaded from the normalized
    patterncir-with-graph JSON instead of the raw captions file. The graph
    JSON is a superset of cap.rc2.{split}.json (same pairid / reference /
    target_hard / target_soft / img_set / caption fields) plus a 'graph'
    field per entry, so downstream consumers that read from self.triplets
    work unchanged and can pick up entry['graph'] if they want it.
    """

    def __init__(self, split: str, mode: str, preprocess: callable,
                 with_graph: bool = False):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.with_graph = with_graph

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        if with_graph:
            triplets_path = patterncir_graph_path / split / f'cap.rc2.{split}.json'
        else:
            triplets_path = patternnet_dataset_path / 'captions' / f'cap.rc2.{split}.json'
        with open(triplets_path) as f:
            self.triplets = json.load(f)

        # name_to_path covers every split-scoped image_splits file so that val
        # captions referencing images stored under train/ or test1/ resolve.
        self.name_to_path = {}
        for s in ('train', 'val', 'test1'):
            with open(patternnet_dataset_path / 'image_splits' / f'split.rc2.{s}.json') as f:
                for name, rel in json.load(f).items():
                    self.name_to_path[name] = patternnet_dataset_path / s / rel

        # Classic mode iterates over this split's image_splits keys.
        with open(patternnet_dataset_path / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.image_names = list(json.load(f).keys())

        print(f"PatternCIR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image = self.preprocess(PIL.Image.open(self.name_to_path[reference_name]))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image = self.preprocess(PIL.Image.open(self.name_to_path[target_hard_name]))
                    return reference_image['pixel_values'], target_image['pixel_values'], rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image = self.preprocess(PIL.Image.open(self.name_to_path[image_name]))
                return image_name, image['pixel_values']

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
