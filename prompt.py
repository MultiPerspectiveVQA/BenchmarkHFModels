from tqdm import tqdm

SIMPLE_STD = 'Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image?'
MULTI_ANS_STD = 'Indicate every possible answer to the given question. Question: {question}'
SIMPLE_COT = 'Please answer the following question by reasoning step by step. Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image?'
MULTI_ANS_COT = 'Please indicate every possible answer to the visual question by reasoning step by step. Question: {question}'

def simple_prompt(dataset, prompt_type):
    if prompt_type == 'std':
        prompt = SIMPLE_STD
    elif prompt_type == 'img-cap':
        pass
    elif prompt_type == 'cot':
        prompt = SIMPLE_COT
    else:
        raise Exception('Invalid prompt type')
    prompt_col = list()
    for record in tqdm(dataset):
        question, answer = record['question'], ', '.join(record['answers'])
        prompt = prompt.format(question=question, answer=answer)
        prompt_col.append(prompt)
    dataset = dataset.add_column('prompt', prompt_col)
    return dataset

def multi_ans_prompt(dataset, prompt_type):
    if prompt_type == 'std':
        prompt = MULTI_ANS_STD
    elif prompt_type == 'img-cap':
        pass
    elif prompt_type == 'cot':
        prompt = MULTI_ANS_COT
    else:
        raise Exception('Invalid prompt type')
    print('Building prompts')
    prompt_col = list()
    for record in tqdm(dataset):
        question, answer = record['question'], ', '.join(record['answers'])
        prompt = prompt.format(question=question, answer=answer)
        prompt_col.append(prompt)
    dataset = dataset.add_column('prompt', prompt_col)
    return dataset

def append_prompts(dataset, test_type, prompt_type):
    if test_type == 'simple':
        return simple_prompt(dataset, prompt_type)
    elif test_type == 'multi-ans':
        return multi_ans_prompt(dataset, prompt_type)
    else:
        raise Exception('Invalid test type')
