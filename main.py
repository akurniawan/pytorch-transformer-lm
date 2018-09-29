import torch.optim as optim

from data import wikitext2
from models.params import RNNLanguageModelParams
from models.encoder import RNNEncoder
from models.rnn import RNNLanguageModel
from models.utils import repackage_hidden

from criterion import lm_criterion


def main(batch_size=20, bptt_len=70, lr=30.):
    wt2 = wikitext2(batch_size=batch_size, bptt_len=bptt_len)

    params = RNNLanguageModelParams(vocab_size=len(wt2.text_field.vocab))
    encoder = RNNEncoder(params)
    model = RNNLanguageModel(encoder=encoder, hyperparams=params)
    hidden = model.init_hidden_layer(batch_size)
    criterion = lm_criterion(params)
    parameters = (list(model.parameters()) + list(criterion.parameters()))
    opt = optim.SGD(parameters, lr=lr)

    for t in wt2.train_iter:
        opt.zero_grad()

        text = t.text
        target = t.target

        out, hidden = model(text, hidden)
        raw_loss = criterion(out.view(-1, out.size(2)), target.view(-1))
        loss = raw_loss[1]

        loss.backward()
        opt.step()

        hidden = repackage_hidden(hidden)


if __name__ == '__main__':
    main()