# Python libraries
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm


def get_results(dataset):
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    results = []
    for record in tqdm(dataset):
        output = [record['image_id'], record['question_id'], record['image_filename'], record['binary_label'], record['question'], '| '.join(record['answers']), record['prompt']]
        inputs = processor(record['image'], text=record['prompt'], return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        output.append(generated_text)
        results.append(output)
    return results