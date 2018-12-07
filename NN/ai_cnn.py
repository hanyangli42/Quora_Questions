import time
import torch
from torch import nn
from torch.nn import functional as func
from torch.autograd import Variable
import pickle
from sklearn.metrics import accuracy_score
from utils import Dataset, get_script_short_name

hparams = {
    'learning_rate': 0.001,
    'max_epoch': 10,
    'display_step': 1000,
    'emb_dim': 300,
    'conv_win': 3,
    'repr_dim': 300,
    'fc0_dim': 300,
    'fc1_dim': 300,
    'n_classes': 2,
    'batch_size': 100
}


class CNN(nn.Module):
    def __init__(self, emb_layer):
        super(CNN, self).__init__()
        self.emb_layer = emb_layer
        self.cnn_layer = nn.Conv1d(hparams['emb_dim'], hparams['repr_dim'], hparams['conv_win'],
                                   padding=int(hparams['conv_win'] / 2))

    def forward(self, x):
        embs = self.emb_layer(x)
        conv = func.tanh(self.cnn_layer(embs.permute(0, 2, 1)))
        return conv.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, emb_layer):
        super(Model, self).__init__()
        self.cnn = CNN(emb_layer)
        self.fc0 = nn.Linear(hparams['repr_dim'] * 2, hparams['fc0_dim'])
        self.fc1 = nn.Linear(hparams['fc0_dim'] * 2, hparams['fc1_dim'])
        self.fc2 = nn.Linear(hparams['fc1_dim'], hparams['n_classes'])
        self.att = nn.Linear(hparams['repr_dim'] + hparams['fc0_dim'], 1)

    def forward(self, q1, q2, q1_len, q2_len):
        r1 = self.cnn(q1)
        r2 = self.cnn(q2)
        h1s, h2s = [], []
        for i in range(len(q1_len)):
            r1i = r1[i, :q1_len[i]]
            r2i = r2[i, :q2_len[i]]
            r1i_size = r1i.size()
            r2i_size = r2i.size()
            joint_size = torch.Size((r1i_size[0], r2i_size[0], r2i_size[1]))
            r1ie = r1i.unsqueeze(1).expand(joint_size)
            r2ie = r2i.unsqueeze(0).expand(joint_size)
            joint = torch.cat([(r1ie - r2ie).abs(), r2ie * r1ie], 2)
            joint = func.tanh(self.fc0(joint.view(-1, joint.size()[-1])))
            joint = joint.view(joint_size[0], joint_size[1], hparams['fc0_dim'])
            s1, _ = joint.max(1)
            s2, _ = joint.max(0)
            s1 = s1.squeeze(1)
            s2 = s2.squeeze(0)
            a1 = self.att(torch.cat([r1i, s1], 1))
            a2 = self.att(torch.cat([r2i, s2], 1))
            a1 = func.softmax(a1.t()).t().expand_as(s1)
            a2 = func.softmax(a2.t()).t().expand_as(s2)
            h1 = (s1 * a1).sum(0).squeeze()
            h2 = (s2 * a2).sum(0).squeeze()
            h1s.append(h1)
            h2s.append(h2)
        r1 = torch.stack(h1s)
        r2 = torch.stack(h2s)
        joint = torch.cat((r1, r2), dim=1)
        joint = torch.tanh(self.fc1(joint))
        out = self.fc2(joint)
        return out


def run_batch(b_data, b_lengths, model, optimizer=None):
    q1 = Variable(torch.from_numpy(b_data['q1']).cuda())
    q2 = Variable(torch.from_numpy(b_data['q2']).cuda())
    outputs = model(q1, q2, b_lengths['q1'], b_lengths['q2'])
    if optimizer:
        y = Variable(torch.from_numpy(b_data['y']).cuda())
        optimizer.zero_grad()
        loss = func.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()
        return loss.data[0]
    else:
        _, predicted = outputs.data.max(1)
        prob = func.softmax(outputs).data
        return predicted, prob[:, 1]


def run_epoch_eval(dataset, model, output_file=''):
    all_plabels, all_pscores = [], []
    batches, batch_lengths = dataset.get_batches(hparams['batch_size'], ('q1', 'q2', 'y'))
    for b_data, b_lengths in zip(batches, batch_lengths):
        plabels, pscores = run_batch(b_data, b_lengths, model)
        all_plabels.extend(plabels.cpu().numpy().tolist())
        all_pscores.extend(pscores.cpu().numpy().tolist())
    if output_file:
        with open(output_file, 'w') as f:
            for s in all_pscores:
                f.write(f'{s:.6f}\n')
    return accuracy_score(dataset.get_data_item('y'), all_plabels)


def run():
    print('Loading data...')
    with open('data/data_emb', 'rb') as f:
        all_sets, embs, word2idx = pickle.load(f)
    emb_layer = nn.Embedding(embs.shape[0], embs.shape[1])
    emb_layer.weight = nn.Parameter(torch.from_numpy(embs))
    model = Model(emb_layer).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    train_set = Dataset(all_sets[0], shuffle=True, pad_keys=('q1', 'q2'))
    dev_set = Dataset(all_sets[1], shuffle=False, pad_keys=('q1', 'q2'))
    test_set = Dataset(all_sets[2], shuffle=False, pad_keys=('q1', 'q2'))
    step = 0
    sum_loss = 0
    dev_best = 0
    test_score = 0
    print("Starting training...")
    print(hparams)
    start_time = time.time()
    for epoch in range(hparams['max_epoch']):
        batches, batch_lengths = train_set.get_batches(hparams['batch_size'], ('q1', 'q2', 'y'))
        for b_data, b_lengths in zip(batches, batch_lengths):
            sum_loss += run_batch(b_data, b_lengths, model, optimizer)
            step += 1
            if step % hparams['display_step'] == 0:
                avg_loss = sum_loss / hparams['display_step']
                sum_loss = 0
                dev_score = run_epoch_eval(dev_set, model)
                out_str = f'Epoc {epoch} iter {step} took {time.time() - start_time:.1f}s\n' \
                          f'loss:\t{avg_loss:.5f}\tdev score:\t{dev_score:.4f}'
                if dev_score > dev_best:
                    dev_best = dev_score
                    output_file = f'pred/{get_script_short_name(__file__)}.pred'
                    test_score = run_epoch_eval(test_set, model, output_file)
                    out_str += f'\t*** New best dev ***\ttest score:\t{test_score:.4f}'
                print(out_str)
                start_time = time.time()
    print('Best model on dev: dev:{:.4f}\ttest:{:.4f}'.format(dev_best, test_score))


if __name__ == '__main__':
    run()
