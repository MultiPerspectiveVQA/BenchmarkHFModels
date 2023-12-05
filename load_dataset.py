# Python libraries
from datasets import load_dataset

# package files
import constants


def load_vqa_therapy(split):
    if split == 'train':
        dataset = load_dataset("imagefolder", data_dir=constants.VQA_THERAPY_TRAIN, split="train")
    elif split == 'val':
        dataset = load_dataset("imagefolder", data_dir=constants.VQA_THERAPY_VAL, split="validation")
    else:
        raise Exception('Incorrect split value')
    return dataset