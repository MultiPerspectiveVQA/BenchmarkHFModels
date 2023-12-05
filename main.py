# Python libraries
import argparse

# package files
from load_dataset import load_vqa_therapy

def main(args):
    dataset = load_vqa_therapy(args.split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Module to help prompt foundation models from hugging face on multi-perspective vqa'
        )
    parser.add_argument('--test-type', choices=['simple', 'multi-ans'], required=True, help='Choose between yes/no style or generating all possible answers')
    parser.add_argument('--model', choices=['blip2', 'instruct-blip'], required=True, help='Choose huggingface model. Blip/InstructBlip')
    parser.add_argument('--prompt-type', choices=['std', 'img-cap', 'cot'], required=True, help='Choose prompt type between standard, image caption as context, and chain of thought')
    parser.add_argument('--split', choices=['train','val'], required=True, help='vqa_therapy dataset split. train/val')
    parser.add_argument('--output-filename', required=True, type=str, help='Filename to store the results. Results will be stored in outputs dir')
    args = parser.parse_args()
    main(args)