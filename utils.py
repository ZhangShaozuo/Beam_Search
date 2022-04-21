from d2l import torch as d2l
import math
import collections
import matplotlib.pyplot as plt
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    source_len = [len(s) for s in xlist]
    target_len = [len(t) for t in ylist]
    src_avg = sum(source_len)/len(source_len)
    tgt_avg = sum(target_len)/len(target_len)
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.xlim(0,30)
    # d2l.plt.text(f'Avg Length: source: {src_avg}, target: {tgt_avg}')

def bleu(pred_seq, label_seq, k):
    if type(pred_seq)==list:
        bleus = []
        for seq in pred_seq:
            bleus.append(_bleu(seq, label_seq, k))
        return max(bleus)
    else:
        return _bleu(pred_seq, label_seq, k)

def _bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# class MyAnimator(d2l.Animator):
#     def save(self, fn = 'Seq2SeqLoss.png'):
#         plt.savefig(self.fig, fn)