# Python libraries
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm


def gen_image_captions(dataset, img_caption_type):
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    caption_col = list()
    for record in tqdm(dataset):
        inputs = processor(record['image'], return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        caption_col.append(generated_text)
    dataset = dataset.add_column('caption', caption_col)
    return dataset
        
