from d2l import torch as d2l
import torch
import torch.nn.functional as F
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X) # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        # print("Encoder: \nShape after embedding: {}".format(X.shape))
        X = X.permute(1, 0, 2) # In RNN models, the first axis corresponds to time steps
        # print("Shape after permute: {}".format(X.shape))
        output, state = self.rnn(X) # When state is not mentioned, it defaults to zeros
        # print("Shape of output/state: {}/{}".format(output.shape, state.shape))
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = torch.nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2) # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        # print("Decoder: \nShape after embed and permute: {}".format(X.shape))
        context = state[-1].repeat(X.shape[0], 1, 1) # Broadcast `context` so it has the same `num_steps` as `X`, `context` shape: (`num_steps`,`batch_size`,`num_hiddens`) ##Why state[-1]
        # print("Shape of context after repeat: {}".format(context.shape))
        X_and_context = torch.cat((X, context), 2) #Concat on the third dim, shape: ((`num_steps`,`batch_size`,`embed_size + num_hiddens`))
        output, state = self.rnn(X_and_context, state) # state as the initial state
        # print("Shape of output/state: {}/{}".format(output.shape, state.shape))
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        output = F.log_softmax(output, dim=2) ### to try
        return output, state