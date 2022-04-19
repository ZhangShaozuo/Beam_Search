### 40.614 Metaheuristic Optimization Project 

Zhang Shaozuo 1003756

### Abstract

Beam Search is a widely adopted decoding algorithm for Sequence to Sequence language models.  This project compared various decoding algorithm including greedy search, beam search and a modified version of beam search.

### Introduction

![seq2seq-predict](D:\Career_Work\40.614 Metaheristic_Optimization\Project\seq2seq_Beam_Search\seq2seq-predict.svg)

Fig. 1 RNN encoder-decoder. Encoder(blue), Decoder(blue)

Figure 1 shows a standard model architecture with Recurrent Neural Network(RNN) block to solve Neural Machine Translation(NMT) task. In encoding phase, the variable-length input is encoded into a state(in-between encoder and decoder), therefore the model parameters are trained. In decoding phase, the model  generate the translated sequence token by token until maximum length or <eos> is reached. Beam search take places in decoding phase. 

### Optimization Problem Formulation

Decoding process could be challenging and costly when the target vocabulary is very large(10,000 more). In decoding phase, output sequence is generated token by token, where the current token is dependent on the previous token. 

Let $s$ denote the number of steps of maximum output sequence, $m$ be the number of steps until <eos>, Vocab = {$x_1,x_2,...,x_v$} be the set of target vocabulary, the problem could be formulated as   

$\max\prod_{x_i,x_{i+1}\in Vocab}^{\min(s,m)} P(x_{i+1}|x_i)$ where $x_0=\text{<bos>},x_{last}=<eos>$

###  Dataset and Preprocess

**Training:  Tab-delimited Bilingual English-French Sentence Pairs**

The contains 167,130 pairs of sentences in both languages. One advantage of this dataset is that for some English sequence, there are multiple corresponding French Sequences. This allows the model to learn that there could be many ways to express the same meaning. 

```python
'''Examples'''
# [English]: You know that your English is good when people stop complimenting you on how good your English is.	
# [France]: Vous savez que votre anglais est bon lorsque les gens arrêtent de vous en complimenter.

# [English]: Chinese officials say economic growth has dropped to a three-year low because of the world economy.	
# Les officiels chinois disent que la croissance économique est tombée à son plus bas niveau depuis trois ans en raison de l'économie mondiale.

# [English]: Girls begin puberty around the ages of ten to eleven, and boys around the ages of eleven to twelve.	
# Les filles commencent leur puberté aux environs de dix ou onze ans et les garçons autour de onze ou douze.

# [English]: I can't believe that you aren't at least willing to consider the possibility of other alternatives.	
# [France]: Je n'arrive pas à croire que vous ne soyez pas au moins disposé à envisager d'autres possibilités.

# [English]: I can't believe that you aren't at least willing to consider the possibility of other alternatives.	
# [France]: Je n'arrive pas à croire que vous ne soyez pas au moins disposée à envisager d'autres possibilités.

# [English]: I can't believe that you aren't at least willing to consider the possibility of other alternatives.	
# [France]:Je n'arrive pas à croire que vous ne soyez pas au moins disposés à envisager d'autres possibilités.
```

![TD_dataset](D:\Career_Work\40.614 Metaheristic_Optimization\Project\seq2seq_Beam_Search\TD_dataset.svg)

​																				*Fig2: Visualization of Tab-delimited Bilingual English-French Sentence Pairs*

**Validation: Multi30k**

Seconding training dataset is Multi30k dataset, which contains less sentence pairs. Multi30k provides validation datasets that are used to evaluate the performance in decoding phase. To augment the dataset, Multi30k training set is incorporated in encoder training as well, 

```python
 '''Description'''
 # Training
 # (en) 29000 sentences, 377534 words, 13.0 words/sent
 # (fr) 29000 sentences, 409845 words, 14.1 words/sent
 # Validation
 # (en) 1014 sentences, 13308 words, 13.1 words/sent
 # (fr) 1014 sentences, 14381 words, 14.2 words/sent
```



**Data Preprocess**

Data preprocess take places before training the model. Some methods adopted are

1. **Tokenization: ** Tokenization is necessary as the model takes the input sequence token by token. Embedding vector is also performed on word level, not on sentence level

```python
I am a student. => ['I','am','a','student','.']
```

2. **Variance-Length: ** Set number of steps as $k$. Sequence longer than $k$ will be truncated, shorter than $k$ will be padded to $k$ length.
3. **Build batch: ** A group of sequence pairs form a batch, the model parameters is updated batch by batch



### Training

**Model Architecture:**

```python
EncoderDecoder(
  (encoder): Seq2SeqEncoder(
    (embedding): Embedding(12890, 32)
    (rnn): GRU(32, 32, num_layers=2, dropout=0.1)
  )
  (decoder): Seq2SeqDecoder(
    (embedding): Embedding(21062, 32)
    (rnn): GRU(64, 32, num_layers=2, dropout=0.1)
    (dense): Linear(in_features=32, out_features=21062, bias=True)
  )
)
```

12,890 is the size of source vocabulary, 21,062 is the target vocabulary

**Experiment Setting: **

Embedding size: 32. Length of the word vector

Number of Hidden layers: 32

Batch Size: 32

Number of Steps: 14, maximum length of output sequence

Number of epochs: 100

Loss Function: Masked SoftMax Cross Entropy Loss

![Seq2Seq](D:\Career_Work\40.614 Metaheristic_Optimization\Project\Seq2Seq.png)

​																				*Fig 3: Training Process*

### Evaluation

Multi30k validation set is used for evaluation. Each validation English sequence and its predicted French  Sequence adopts (Bilingual Evaluation Understudy)BLEU Score as the criteria. We obtain BLEU score 1.0 if the prediction and ground truth perfectly match, 0 if completely does not match. 

**Greedy Search**

Given validation sample *src_sentence* and trained model *net*

1. **Encoding: ** textual information into digital vectors

```python
src_tokens = src_vocab[src_sentence.split(' ')] + [src_vocab['<eos>']]
```

2. **Initialization**: First generated token <bos>, initial hidden state *dec_state*
3. **Iteration**: Do iterations for *num_steps* times, in each loop

```python

```



<img src="D:\Career_Work\40.614 Metaheristic_Optimization\Project\evaluate_greedy.png" alt="evaluate_greedy" style="zoom:50%;" />

Average: 0.415, variance: 0.0237

### Solution Encoding and Search Operators

### Summary of the Metaheuristics/ Rules

Beam Search Explanation



### Results

