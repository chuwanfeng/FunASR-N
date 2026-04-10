class TextProcessor:
    def __init__(self, vocab):
        self.vocab = vocab
        self.char2idx = {c:i for i,c in enumerate(vocab)}
        self.idx2char = {i:c for i,c in enumerate(vocab)}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, indices):
        return "".join([self.idx2char.get(i, "") for i in indices])