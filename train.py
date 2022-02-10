import torch
import torch.nn as nn
import torch.optim as optim
import model
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.data.metrics import bleu_score
from torchtext.vocab import build_vocab_from_iterator

writer = SummaryWriter()

def data_process(raw_text_iter, tokenizer, vocab):
    '''Converts raw text into a flat Tensor.'''
    data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) \
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz, device) :
    '''
    Divides the data into bsz seperate sequences, removing extra elemts
    that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tenosr of shape [N // bsz, bsz]
    '''
    seq_len = data.size(0) //bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt) :
    '''
    Args:
        source - Tensor, shape [ full_seq_len, batch_size]
        i - int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size]
        target has shape [seq_len * batch_size]
    '''

    seq_len = min(bptt, len(source) -1 -i)
    data = source[i:i+seq_len]
    target = source[i+1: i+1+seq_len].reshape(-1)
    return data, target

def compute_lr(step_num, optimizer=None):
    lr = (config.dim_model ** -0.5) * \
        min(step_num ** -0.5, step_num * \
                    (config.warmup_steps ** -1.5))
    if optimizer is None:
        return lr
    else :
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train(net, train_data, ntokens, optimizer, bptt, epoch, device):
    net.train()
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    num_batches = len(train_data) // bptt
    tq = tqdm(enumerate(range(0, train_data.size(0) -1, bptt)), \
              desc='train E{:03d}'.format(epoch), ncols = 0)
    for batch, i in tq:
        data, targets = get_batch(train_data, i, bptt)
        data = data.to(device)
        targets = targets.to(device)
        output = net(data,data)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        fmt = '{:.4f}'.format
        tq.set_postfix(mini_batch_loss = fmt(loss))
        global step_num
        step_num += 1
        compute_lr(step_num, optimizer)
        writer.add_scalar("Loss/train", loss, step_num-1)

def evaluate(net, eval_data, bptt, epoch, device):
    net.eval()
    tq = tqdm(enumerate(range(0, train_data.size(0)-1, bptt)), \
              desc='eval E{:03d}'.format('eval',epoch), ncols =0)
    with torch.no_grad():
        for batch, i in tq:
            data, targets = get_batch(train_data, i)
            data = data.to(device)
            targets = targets.to(device)

            out = net(data,data)
            score = bleu_score(out, targets).cpu()
            fmt = '{:.4f}'.format
            tq.set_postfix(bleu-score(score))
            writer.add_scalar("BLeu score", score)

def main():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter),\
                                    specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, tokenizer, vocab)
    val_data = data_process(val_iter, tokenizer, vocab)
    test_data = data_process(test_iter, tokenizer, vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('{} available'.format(device))

    batch_size = 20
    eval_batch_size = 10

    train_data = batchify(train_data, batch_size,device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data =batchify(test_data, eval_batch_size, device)

    bptt = 35

    ntokens = len(vocab)
    net = model.Transformer(ntokens).to(device)

    global step_num
    step_num= 1
    initial_lr = compute_lr(step_num)
    optimizer = torch.optim.Adam([p for p in net.parameters() \
                                  if p.requires_grad], \
                                 lr = initial_lr, betas = (0.9,0.98), \
                                 eps = 1e-09)
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, config.epochs+1):
        train(net, train_data, ntokens, optimizer, bptt, epoch, device)
        evaluate(net, eval_data, bptt, epoch, device)


if __name__ == '__main__':
    main()

