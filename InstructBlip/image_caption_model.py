# Python libraries
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from tqdm import tqdm

SIMPLE_CAP = "A photo of"
Q_GUIDED_CAP = "Describe this image according to the given question: {question}"

def gen_image_captions(dataset, img_caption_type):
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    caption_col = list()
    for record in tqdm(dataset):
        if img_caption_type == 'std':
            prompt = SIMPLE_CAP
        elif img_caption_type == 'guided':
            prompt = Q_GUIDED_CAP
        else:
            raise Exception('Invalid image caption type')
        inputs = processor(record['image'], text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        caption_col.append(generated_text)
    dataset = dataset.add_column('caption', caption_col)
    return dataset
        
