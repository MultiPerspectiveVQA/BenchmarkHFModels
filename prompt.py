from tqdm import tqdm

from Blip2.image_caption_model import gen_image_captions as gen_blip2_image_captions

SIMPLE_STD = 'Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image?'
SIMPLE_IMG_CAP = 'Context: {caption} Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image?'
SIMPLE_COT = 'Please answer the following question by reasoning step by step. Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image?'

MULTI_ANS_STD = 'Indicate every possible answer to the given question. Question: {question}'
MULTI_ANS_IMG_CAP = 'Context: {caption} Indicate every possible answer to the given question. Question: {question}'
MULTI_ANS_COT = 'Please indicate every possible answer to the visual question by reasoning step by step. Question: {question}'


def gen_simple_qa_prompt(record, template):
    prompt = template
    question, answer = record['question'], ', '.join(record['answers'])
    prompt = prompt.format(question=question, answer=answer)
    return {'prompt': prompt}

def gen_simple_cap_prompt(record, template):
    prompt = template
    caption, question, answer = record['caption'], record['question'], ', '.join(record['answers'])
    prompt = prompt.format(question=question, answer=answer)
    return {'prompt': prompt}

def gen_multi_ans_qa_prompt(record, template):
    prompt = template
    question = record['question']
    prompt = prompt.format(question=question)
    return {'prompt': prompt}

def gen_multi_ans_cap_prompt(record, template):
    prompt = template
    caption, question = record['caption'], record['question']
    prompt = prompt.format(caption=caption, question=question)
    return {'prompt': prompt}

def gen_img_caption(dataset, model, img_caption_type):
    if model == 'blip2':
        dataset = gen_blip2_image_captions(dataset, img_caption_type)
    elif model == 'instruct_blip':
        pass
    else:
        raise Exception('invalid model/ image caption type')
    return dataset

def append_prompts(dataset, args):
    if args.test_type == 'simple' and args.prompt_type == 'std':
        return dataset.map(gen_simple_qa_prompt, batched=False, fn_kwargs={'template': SIMPLE_STD})
    elif args.test_type == 'simple' and args.prompt_type == 'img_cap':
        dataset = gen_img_caption(dataset, args.model, args.img_caption_type)
        return dataset.map(gen_simple_cap_prompt, batched=False, fn_kwargs={'template': SIMPLE_IMG_CAP})
    elif args.test_type == 'simple' and args.prompt_type == 'cot':
        return dataset.map(gen_simple_qa_prompt, batched=False, fn_kwargs={'template': SIMPLE_COT})
    elif args.test_type == 'multi_ans' and args.prompt_type == 'std':
        return dataset.map(gen_multi_ans_qa_prompt, batched=False, fn_kwargs={'template': MULTI_ANS_STD})
    elif args.test_type == 'multi_ans' and args.prompt_type == 'img_cap':
        dataset = gen_img_caption(dataset, args.model, args.img_caption_type)
        return dataset.map(gen_multi_ans_qa_prompt, batched=False, fn_kwargs={'template': MULTI_ANS_IMG_CAP})
    elif args.test_type == 'multi_ans' and args.prompt_type == 'cot':
        return dataset.map(gen_multi_ans_qa_prompt, batched=False, fn_kwargs={'template': MULTI_ANS_COT})
    else:
        raise Exception('Invalid test type/ prompt type')
