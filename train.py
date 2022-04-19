import torch
from d2l import torch as d2l
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
from model import *
from preprocess import *
from utils import *

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-embed_size', type = int, default = 32,
                   help = 'Embedding size of Seq2Seq')
    p.add_argument('-num_hiddens', type = int, default = 32,
                   help = 'Number of hidden layers')
    p.add_argument('-num_layers', type = int, default = 2,
                   help = 'Number of layers')
    p.add_argument('-dropout', type = float, default = 0.1,
                   help = 'Dropout rate')
    p.add_argument('-batch_size', type = int, default = 32,
                   help = 'Batch size')
    p.add_argument('-num_steps', type = int, default = 14,
                   help = 'Number of steps')
    p.add_argument('-lr', type = float, default = 0.005,
                   help = 'Learning rate')
    p.add_argument('-num_epochs', type = int, default = 100,
                   help = 'Number of epoches')
    p.add_argument('-num_samples', type = int, default = None,
                   help = 'Number of training samples')
    p.add_argument('-output_p', type = str, default = 'Seq2Seq.pt',
                   help = 'Path of trained model')
    p.add_argument('-data_aug', type = bool, default = True,
                   help = 'Whether to perform data augmentation')
    p.add_argument('-interval', type = int, default = 10,
                   help = 'Animator epoch interval')
    # p.add_argument('-device', type = int, default = dl2.try_gpu(),
    #                help = 'Number of epoches')
    return p.parse_args()

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    num_steps = X.size(1)
    # print(torch.arange(num_steps, dtype=torch.float32,
    #                     device=X.device)[None,:])
    mask = torch.arange(num_steps, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(torch.nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).sum(dim=1) ### try sum
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device, interval):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == torch.nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                         xlim=[10, num_epochs])
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[interval, num_epochs])
    for epoch in tqdm(range(num_epochs)):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % interval == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss: {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
    animator.fig.savefig('Seq2Seq.png')

def main():
    args = parse_arguments()
    # raw_text = read_data_nmt('fra.txt') # len: 11489286
    # text = preprocess_nmt(raw_text) # len: same as raw_text
    # source, target = tokenize_nmt(text, num_examples=args.num_samples)
    embed_size, num_hiddens, num_layers, dropout = args.embed_size, args.num_hiddens, args.num_layers, args.dropout
    batch_size, num_steps = args.batch_size, args.num_steps
    lr, num_epochs, device = args.lr, args.num_epochs, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps ,num_examples=args.num_samples, fn = 'fra.txt', da = args.data_aug)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, interval = args.interval)
    torch.save(net, args.output_p) ### save the entire model

if __name__ == '__main__':
    main()