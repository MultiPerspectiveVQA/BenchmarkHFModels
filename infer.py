from Blip2.inference import get_results as blip2_results
from InstructBlip.inference import get_results as instruct_blip_results

def get_results(dataset, model):
    if model == 'blip2':
        return blip2_results(dataset)
    elif model == 'instruct_blip':
        return instruct_blip_results(dataset)
    else:
        raise Exception('Invalid model type')