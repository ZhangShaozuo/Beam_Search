from preprocess import *
from utils import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l
import math
import pickle
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False, method = 'beam_search', beam_size = 2):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    if method == 'beam_search':
        decode = {0:{'dec_Xs':[], 'preds':[], 'scores':[]}}
        
        Y, dec_state = net.decoder(dec_X, dec_state)
        decode[0]['Ys']=Y
        decode[0]['dec_states']=dec_state
        prob_a = sum([math.exp(y) for y in Y.squeeze().detach()])

        decode[0]['dec_Xs']= torch.topk(Y,beam_size).indices
        # print(decode[0]['dec_Xs'][:,:,0], decode[0]['dec_Xs'])
        decode[0]['preds'] = decode[0]['dec_Xs'][0,0].detach()
        for pred in decode[0]['preds']:
            score = Y[:,:,pred].squeeze(dim=0).type(torch.float32).item()
            score = math.log(math.exp(score)/prob_a)
            decode[0]['scores'].append(score)
        
        # Yc = Y.clone()
        # for i in range(beam_size):
        #     dec_X = Yc.argmax(dim=2)
        #     pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        #     score = Y[:,:,pred].squeeze(dim=0).type(torch.float32).item()
        #     score = math.log(math.exp(score)/prob_a)
        #     decode[0]['dec_Xs'].append(dec_X)
        #     decode[0]['preds'].append(pred)
        #     decode[0]['scores'].append(score)
        #     Yc[:,:,pred] = -999
        # print(decode)
        wip = [1] * beam_size
        # num_steps = 4
        for step in range(1,num_steps):
            print(step)
            decode[step]={}
            ranking = {}
            for i in range(beam_size):
                decode[step][i] = {}
                if wip[i] == 1:
                    if step==1:
                        dec_state = decode[step-1]['dec_states']
                        dec_X = decode[step-1]['dec_Xs'][:,:,i]
                    else:
                        # print(f"step,i,k: {step, i}")
                        # print(decode[step-1])
                        # print(tgt_vocab['<eos>'])
                        dec_state = decode[step-1][i]['dec_states']
                    Y, dec_state = net.decoder(dec_X, dec_state)

                    prob_a = sum([math.exp(y) for y in Y.squeeze().detach()])
                    Yc = Y.clone()
                    
                    decode[step][i]['Ys']=Y
                    decode[step][i]['dec_states']=dec_state

                    for k in range(beam_size):
                        decode[step][(i,k)]={}
                        dec_X = Yc.argmax(dim=2)
                        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
                        if pred == tgt_vocab['<eos>']:
                            print(f"step,i,k: {step, i, k} reached the end")
                            break

                        decode[step][(i,k)]['dec_Xs']=dec_X
                        
                        cur_score = Y[:,:,pred].squeeze(dim=0).type(torch.float32).item()
                        cur_score = math.log(math.exp(cur_score)/prob_a)
                        if step == 1:
                            pre_score = decode[step-1]['scores'][i]
                            pre_pred = decode[step-1]['preds'][i]
                            # print((i,k), pre_pred)
                            decode[step][(i,k)]['preds']=[pre_pred, pred]
                            # print([pre_pred, pred])
                        else:
                            # print(decode[step-1])
                            pre_score = decode[step-1][i]['scores']
                            pre_pred = decode[step-1][i]['preds']
                            # print((i,k), pre_pred, pred)
                            cur_pred = pre_pred + [pred]
                            decode[step][(i,k)]['preds'] = cur_pred
                            # print(step, (i,k), decode[step][(i,k)]['preds'])
                        decode[step][(i,k)]['scores']=cur_score+pre_score
                        ranking[(i,k)] = cur_score+pre_score
                        Yc[:,:,pred] = -999 
            
            s_ranking = [k for k, _ in sorted(ranking.items(), key=lambda item: item[1])][:-beam_size]
            for p in s_ranking:
                del decode[step][p]
            decode[step] = reformat(decode[step])
            # print(f'ranking: {ranking}')
            # print(decode[step])
            # print(decode[step].keys())
    else:
        for _ in range(num_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            # We use the token with the highest prediction likelihood as the input
            # of the decoder at the next time step
            dec_X = Y.argmax(dim=2)
            pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weight_seq.append(net.decoder.attention_weights)
            # Once the end-of-sequence token is predicted, the generation of the
            # output sequence is complete
            if pred == tgt_vocab['<eos>']:
                break
            output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def reformat(decode):
    op = {}
    for i,k in enumerate(decode.keys()):
        if type(k) == tuple:
            op[i-1]={}
            b1,b2 = k
            op[i-1].update({'Ys':decode[b1]['Ys']})
            op[i-1].update({'dec_states':decode[b1]['dec_states']})
            op[i-1].update(decode[k])
    return op
def main():
    en = read_data_nmt('val.lc.norm.tok.en')
    engs = en.split('\n')
    fr = read_data_nmt('val.lc.norm.tok.fr')
    fras = fr.split('\n')
    net = torch.load('Seq2Seq.pt')
    net.eval()
    device = d2l.try_gpu()
    num_steps = 14
    src_vocab = torch.load('src_vocab.pth')
    tgt_vocab = torch.load('tgt_vocab.pth')
    bleu_dist = []
    for eng, fra in zip(engs[:1], fras[:1]):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, method='beam_search')
        # print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=1):.3f}')
        bleu_dist.append(bleu(translation, fra, k=1))
    # plt.hist(bleu_dist, bins=100)
    # plt.gca().set(title='BLEU Score Frequency Histogram: Greedy Decode', ylabel='Frequency',xlabel='BLEU Score')
    # plt.savefig('evaluate.png')
    print(f'Average {round(sum(bleu_dist)/len(bleu_dist),3)}, variance {np.var(bleu_dist)}')

if __name__=='__main__':
    main()