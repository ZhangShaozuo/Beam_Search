import os
import torch
from d2l import torch as d2l
# import math
# import collections

def read_data_nmt(fn = 'fra.txt'):
    '''Open and read the text file'''
    with open(os.path.join(fn), 'r', encoding = 'utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    '''Preprocess the Any-English dataset.'''
    def no_space(char, prev_char):
        '''Check whether it is a sign and there is no space before'''
        return char in set(',.!?') and prev_char != ' '

    '''Replace non-breaking space with space, and convert uppercase letters to lowercase ones'''
    '''Note that Chinese characters don't have space in between'''
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    '''Insert space between words and punctuation marks'''
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """Tokenize the Chinese-English dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        ## checkpoints
        # if i<5:
        #     print(f'line: {line}')
        #     print(f'parts: {parts}')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
            '''If target language is Chinese, then len(parts)==3,
            but since Chinese works very bad'''
            # target.append(list(parts[1]))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=None, fn = 'fra.txt', da = True):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt(fn))
    source, target = tokenize_nmt(text, num_examples)
    print("Source Len: {}, Target Len: {}".format(len(source), len(target)))
    if da == True:
        engs = read_data_nmt('train.lc.norm.tok.en').split('\n')
        fras = read_data_nmt('train.lc.norm.tok.fr').split('\n')
        engs = [eng.split(' ') for eng in engs]
        fras = [fra.split(' ') for fra in fras]
        source.extend(engs)
        target.extend(fras)
        print(source[-3:],target[-3:])
    print("Source Len: {}, Target Len: {}".format(len(source), len(target)))
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    torch.save(src_vocab, 'src_vocab.pth')
    torch.save(tgt_vocab, 'tgt_vocab.pth')
    print("Source Vocab Size: {}, Target Vocab Size: {}".format(len(src_vocab), len(tgt_vocab)))
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    print("Source Array Shape: {}, Target Array Shape: {}".format(src_array.shape, tgt_array.shape))
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab