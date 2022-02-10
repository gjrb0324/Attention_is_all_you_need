#model.py
#Model - 1) Encoder - 1. Multi-Head(self), 2.Feed Forward
#        2) Decoder - 1. Multi-Head(self), 2.Multi-Head(with output of encoder)
#                     , 3. Feed Forward
import math
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import config

class Transformer(nn.Module):
    def __init__(self,num_tokens):
        '''
        target_word_num : number of possible target words
        '''
        super().__init__()
        self.linear = nn.Linear(config.dim_model, num_tokens)
        init.xavier_uniform_(self.linear.weight)
        global shared_weight
        shared_weight = self.linear.weight

        #Share same weight matrix between two embedding layers
        #and the pre-softmax linear transformation
        self.encoder = Encoder(num_tokens)
        self.decoder = Decoder(num_tokens)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,encoded_input, encoded_output):
        encoder_out = self.encoder(encoded_input)
        decoder_out = self.decoder(encoded_output, encoder_out)
        out = self.linear(decoder_out)
        out = self.softmax(out)
        return out


class Encoder(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.ffn = Feed_Forward()
        self.mha = Multi_head(0)

        #Sharing embedding weight with linear fc's weight
        self.embedding = nn.Embedding(num_tokens,config.dim_model)
        global shared_weight
        self.embedding.weight = shared_weight
        self.positional = PositionalEncodding()
        self.layer_norm = nn.LayerNorm(config.dim_model)

    def forward(self,seq):
        #prev_seq : previous output sequence from Nth encoder layer
        #First prev_seq ; positional encoded (embedded) seq
        prev_seq = self.embedding(seq)
        prev_seq = self.positional(prev_seq)

        for i in range(config.num_recur):
            mha_out = self.mha(prev_seq)
            mha_out = self.layer_norm(mha_out + prev_seq)
            ffn_out = self.ffn(mha_out)
            ffn_out = self.layer_norm(mha_out + ffn_out)
            prev_seq = ffn_out

        result = prev_seq
        return result


class Decoder(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.ffn = Feed_Forward()
        self.mha1 = Multi_head(1)
        self.mha2 = Multi_head(2)

        #Sharing Embedding weight with encoder's embedidng weight, linear fc's weight
        self.embedding = nn.Embedding(num_tokens, config.dim_model)
        global shared_weight
        self.embedding.weight = shared_weight

        self.positional = PositionalEncodding()
        self.layer_norm = nn.LayerNorm(config.dim_model)

    def forward(self, seq, encoder_out):
        #parameters:    encoder_out - output of encoder
        #prev_seq : previous output sequence from Nth encoder layer
        #First prev_seq ; positional encoded (embedded) seq
        prev_seq = self.embedding(seq)
        prev_seq = self.positional(prev_seq)

        for i in range(config.num_recur):
            mha1_out = self.mha1(prev_seq)
            mha1_out = self.layer_norm(prev_seq+mha1_out)
            mha2_out = self.mha2(mha1_out, encoder_out)
            mha2_out = self.layer_norm(mha2_out + mha1_out)
            ffn_out = self.ffn(mha2_out)
            ffn_out = self.layer_norm(ffn_out + mha2_out)
            prev_seq = ffn_out

        result = prev_seq
        return result

class Multi_head(nn.Module):
    def __init__(self,mode):
        super().__init__()
        '''
        There are three types of multi head-attention,
        0. Self(Encoder), 1. Self(Decoder), 2.Encoder_Decoder
        Those number above are given through 'mode' variable.
        '''
        need_mask = True if mode==1 else False
        self.mode = mode
        self.scaled_dot = Scaled_Dot_Product(dec_dec=need_mask)
        self.linear = nn.Linear(config.num_head * config.dim_v, \
                                config.dim_model)

        self.Wq = torch.nn.Parameter(torch.zeros(config.num_head, \
                                                 config.dim_model,\
                                                 config.dim_k), \
                                     requires_grad =True)
        self.Wk = torch.nn.Parameter(torch.zeros(config.num_head, \
                                                 config.dim_model,\
                                                 config.dim_k), \
                                     requires_grad = True)
        self.Wv = torch.nn.Parameter(torch.zeros(config.num_head, \
                                                 config.dim_model,\
                                                 config.dim_v), \
                                     requires_grad = True)

        init.xavier_uniform_(self.Wq.data)
        init.xavier_uniform_(self.Wk.data)
        init.xavier_uniform_(self.Wv.data)
        init.xavier_uniform_(self.linear.weight)

    def forward(self, prev_seq, encoder_out = None):
        '''
        Parameter: prev_seq, encoder_out - Tensor,
        shape [batch_num, batch_size(=num_words in a batch), dim_v]
        '''
        if self.mode == 2:
            if encoder_out is not None:
                #num_batch : The # of batches
                num_batch =  encoder_out.size(0)
            else:
                sys.exit('\n The result from encoder is not given while mode' +
                         'is set by encoder-decoder multi-head')
        else:
            num_batch = prev_seq.size(0)

        result = torch.tensor([])
        for j in range(num_batch):
            concatted = torch.tensor([])
            for i in range(config.num_head):
                q =  torch.mm(prev_seq[j,:,:], self.Wq[i,:,:])
                if self.mode == 2:
                    if encoder_out is not None:
                        k = torch.mm(encoder_out[j,:,:],self.Wk[i,:,:])
                        v = torch.mm(encoder_out[j,:,:] ,self.Wv[i,:,:])
                    else:
                        sys.exit('\n The result from encoder is not given \
                                 while the mode is set by encoder-decoder \
                                 multi-head')

                else :
                    k = torch.mm(prev_seq[j,:,:],self.Wk[i,:,:])
                    v = torch.mm(prev_seq[j,:,:],self.Wv[i,:,:])
                concatted = torch.cat((concatted, self.scaled_dot(q,k,v)),\
                                      dim=1)
            result  = torch.cat((result, self.linear(concatted).unsqueeze(0)),\
                                dim=0)

        return result


class Scaled_Dot_Product(nn.Module):
    def __init__(self,dec_dec):
        super().__init__()
        self.need_mask = dec_dec
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v):
        out = (torch.mm(q, k.transpose(0,1))/math.sqrt(config.dim_k))
               #matmul & scaling => shape : [dim_encoded, dim_encoded]
        if self.need_mask:
            #Why Masking? : The transformer model only focuses on previous and
            #present words, not coming words.
            #Create masking matrix: from left-below to diagonal: 0, else : -inf
            minus_inf_mask = torch.triu(torch.ones(out.size(0), out.size(0)),\
                                        diagonal=1)
            #Sum out and masking matrix
            out = out + minus_inf_mask
        out = self.softmax(out)
        out = torch.mm(out,v)
        return out



class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.relu =  nn.ReLU(inplace=True)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self,x):
    #Param x : input
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

#Clone Codded from nn.Transformers
class PositionalEncodding(nn.Module):
    def __init__(self,max_len = 5000):
        super().__init__()
        #Apply dropout to tthe sums of embeddign and the positional encodings
        #in both encoder and decoder stacks
        self.dropout =  nn.Dropout(p=0.1)
        position =  torch.arange(max_len).unsqueeze(1) #(Max_length * 1)
        #(1 * dim_model/2)
        div_term = torch.exp(torch.arange(0, config.dim_model, 2) * \
                             (-math.log(10000.0) /config.dim_model))
        pe = torch.zeros(max_len, 1, config.dim_model)
        # => (Max_length * dim_model/2)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        #register_buffer : register buffer which not considered as parameter
        self.register_buffer('pe',pe)


    def forward(self, x) :
        '''
        Shape of param x : [seq_len * batch_size * embedding_dim]
        '''
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


