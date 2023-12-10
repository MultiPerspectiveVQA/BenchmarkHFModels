# Python libraries
import argparse
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm import tqdm


reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)

def update_grd_acc(grd_acc, ground_truth, pred):
    answers = '|'.split(ground_truth)
    total_ans = len(answers)
    count = 0
    for ans in answers:
        reference = [[ans.strip()]]
        candidate = [pred]
        score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        count = count + 1 if score >= 0.5 else count
    grd_acc +=  total_ans / count
    return grd_acc


def eval(args):
    data = pd.read_csv(args.results_file, header=0)
    same_grd_count, diff_grd_count, same_grd_acc, diff_grd_acc = 0, 0, 0, 0
    for _, row in tqdm(data.iterrows()):
        if row['binary_label'] == 'same':
            same_grd_count += 1
            same_grd_acc = update_grd_acc(same_grd_acc, row['answers'], row['result'])
        else:
            diff_grd_count += 1
            diff_grd_acc = update_grd_acc(diff_grd_acc, row['answers'], row['result'])
    overall_accuracy = (same_grd_acc + diff_grd_acc) / (same_grd_count + diff_grd_count)
    same_grd_acc /= same_grd_count
    diff_grd_count /= diff_grd_acc
    print(f'Results')
    print(f'Overall accuracy: {overall_accuracy}')
    print(f'Same grounding accuracy: {same_grd_acc}')
    print(f'Diff grounding accuracy: {diff_grd_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evalue simple outputs',
        description='Module to help evaluate simple format results for MultiPurpose VQA'
        )
    parser.add_argument('--results_file', required=True, type=str, help='Absolute path of the results file')
    args = parser.parse_args()
    eval(args)