import torch
from torch import nn
import nltk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import sklearn
from scipy.stats import mode
import matplotlib.pyplot as plt


def tokenize(text):
    strp = text.strip()
    return nltk.word_tokenize(strp)


def get_data(fname):

    with open(fname, 'r', encoding='utf-8') as f:
        text = f.readlines()

    data = []
    for i in range(0, len(text), 6):
        data.append({
            'source': text[i].strip(),
            'reference': text[i+1].strip(),
            'candidate': text[i+2].strip(),
            'quality': float(text[i+3].strip()),
            'label': text[i+4].strip()
        })

    return data


def get_vocabulary(data_tk):
    vocab = set(["UNK"])
    for row in data_tk:
        vocab.update(row['reference'])
        vocab.update(row['candidate'])

    idx_word = dict(enumerate(vocab))
    word_idx = {v:k for k,v in idx_word.items()}
    return vocab, idx_word, word_idx


def embed(tokens, word_idx, weights_matrix):
    return [weights_matrix[word_idx[tk]] for tk in tokens]


def preprocess(data, embeddings, weights_matrix=None, word_idx=None, add_words=None):
    """ Tokenizes and embeds the reference and candidate sentences. Does not pad those sequences at all though. """
    tokenized = []

    embedding_size = len(embeddings['a'])

    for row in data:
        tokenized.append({
            'reference': tokenize(row['reference']),
            'candidate': tokenize(row['candidate']),
            'quality': row['quality']
        })
    
    vocab_tokenized = []
    
    if add_words is not None:
        for row in add_words:
            vocab_tokenized.append({
                'reference': tokenize(row['reference']),
                'candidate': tokenize(row['candidate']),
                'quality': row['quality']
            })

    if word_idx is None:
    
        vocab, idx_word, word_idx = get_vocabulary(vocab_tokenized)

        # Create an empty matrix for our word embeddings
        vocab_size = len(vocab)
        weights_matrix = np.zeros((vocab_size, embedding_size))

        # Fill the weights matrix with the embeddings from gensim
        words_found = 0
        for i, word in idx_word.items():
            try:
                weights_matrix[i] = embeddings[word]
                words_found += 1
            except KeyError:
                # If the word isn't in our pretrained embeddings, give it a random embedding vector
                weights_matrix[i] = np.random.normal(size=(embedding_size,))
        
    def word_index(w):
        """ Take a word and return the index of an embedding """
        try:
            return word_idx[w]
        except KeyError:
            return word_idx['UNK']

    indexed = []
    for row in tqdm(tokenized):
        indexed.append({
            'reference': [word_index(tk) for tk in row['reference']],
            'candidate': [word_index(tk) for tk in row['candidate']],
            'quality': row['quality']
        })

    y = np.array([int(row['label'] == 'H') for row in data])

    return indexed, y, weights_matrix, word_idx


def get_loader(X,y):
    loadable = []

    for i in range(len(y)):
        seq = (
            i,
            torch.tensor(X[i]['reference']),
            torch.tensor(X[i]['candidate']),
            torch.tensor(X[i]['quality']),
            y[i]
        )
        loadable.append(seq)

    return torch.utils.data.DataLoader(loadable, batch_size=1)
    
class BaselineEstimator(sklearn.base.BaseEstimator):
    """ Define a baseline estimator that guesses the most common class """
    def __init__(self):
        super().__init__()
        self.value = 0
    
    def fit(self, X, y):
        self.value, _ = mode(y)
    
    def predict(self, X):
        return np.full(X.shape[0], self.value)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return (y == y_pred).mean()


class NN(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights_matrix).to('cuda'), freeze=True)
        self.lstm = nn.LSTM(input_size=201, hidden_size=500, batch_first=True)
        self.linear = nn.Linear(500, 2)
        self.activation = lambda x: x

    def forward(self, input):

        z1, _ = self.lstm(input)
        z2 = self.linear(z1[:,-1])

        return self.activation(z2)


def train_model(model, loader, optimizer, epochs=1):
    model.train()
    model.double()
    losses = []

    for epoch in range(epochs):
        loss_total = 0
        for id, ref, cand, q, gold in tqdm(loader):

            ref_embed = model.embedding(ref)
            cand_embed = model.embedding(cand)
            #print(ref_embed.shape)
            #print(cand_embed.shape)
            #print("Try this:", ref_embed.squeeze(0).shape)
            pad = nn.utils.rnn.pad_sequence([ref_embed.squeeze(0), cand_embed.squeeze(0)], batch_first=False)
            pad = pad.reshape(1,-1,100)
            #print("Pad shape:", pad.shape)
            #assert ref_embed.shape == cand_embed.shape

            y_pred = model(pad)

            #print("Model output:", y_pred)
            loss = (y_pred - gold)**2
            loss_total += loss
        #print(loss_total)
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        losses.append(loss_total)
    return losses

def train_epoch(model, loader, optimizer):
    criterion = nn.CrossEntropyLoss()
    loss_total = 0
    for id, ref, cand, q, gold in (loader):

        ref_embed = model.embedding(ref.to('cuda'))
        cand_embed = model.embedding(cand.to('cuda'))
        quality = q.to('cuda')

        pad = nn.utils.rnn.pad_sequence([ref_embed.squeeze(0), cand_embed.squeeze(0)], batch_first=False)
        pad = pad.reshape(1,-1,200)

        # Add the quality values into the sequence
        seq_len = pad.shape[1]
        q = quality.repeat(seq_len)

        pad = torch.cat([pad,q.reshape(1,seq_len,1)], dim=2)

        y_pred = model(pad)

        #loss = (y_pred.to('cpu') - gold)**2
        loss = criterion(y_pred.cpu(), gold)
        loss_total += loss
    
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return loss_total / len(loader)

def test_model(model, loader):
    test_loss = 0
    y_pred = []
    y_true = []
    for id, ref, cand, q, gold in (loader):
        ref_embed = model.embedding(ref.to('cuda'))
        cand_embed = model.embedding(cand.to('cuda'))

        pad = nn.utils.rnn.pad_sequence([ref_embed.squeeze(0), cand_embed.squeeze(0)], batch_first=False)
        pad = pad.reshape(1,-1,200)

        # Add the quality values into the sequence
        seq_len = pad.shape[1]
        q = q.to('cuda').repeat(seq_len)

        pad = torch.cat([pad,q.reshape(1,seq_len,1)], dim=2)

        pred = model(pad)

        y_pred.append(pred[0])
        y_true.append(gold)
    #print(y_pred)
    
    return torch.stack(y_pred), torch.tensor(y_true)

if __name__ == "__main__":

    #references = [len(row['reference']) for row in X]
    #print("Longest reference tokenization:", max(references))
    #candidates = [len(row['candidate']) for row in X]
    #print("Longest candidate tokenization:", max(candidates))

    import gensim.downloader
    embeddings = gensim.downloader.load("glove-wiki-gigaword-100")
    print("Embeddings loaded")

    X, y, embed_weights, _ = preprocess(get_data("data/train.txt"), embeddings)
    print("Preprocessing complete")

    model = NN(embed_weights)
    loader = get_loader(X, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    print("Begin training")
    train_epoch(model, loader, optimizer)


    qualities = np.array([[row['quality']] for row in X])

    X_test, y_test, _, _ = preprocess(get_data("data/test.txt"), embeddings)
    test_qualities = np.array([[row['quality']] for row in X_test])

    model = BaselineEstimator()
    model.fit(qualities,y)
    print(f"Baseline score: {model.score(test_qualities,y_test):.2%}")


    model = LogisticRegression()
    model.fit(qualities,y)
    print(f"Logistic Regression score: {model.score(test_qualities,y_test):.2%}")
