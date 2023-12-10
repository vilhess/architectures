import os
from typing import Any
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image


spacy_eng = spacy.load('eng')


class Vocabulary:
    def __init__(self, freq_treshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_treshold = freq_treshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_treshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_treshold=5):
        super().__init__()
        self.root_dir = 'root_dir'
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_treshold)
        self.vocab.build_vocabulary(self.captions.to_list())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img = Image.open(os.path.join(
            self.root_dir, self.imgs[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        caption = self.captions[index]
        numericalized_captions = [self.voca.stoi["<SOS>"]]
        numericalized_captions += self.vocab.numericalize(caption)
        numericalized_captions.append(self.voca.stoi["<EOS>"])

        return img, torch.Tensor(numericalized_captions)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False,
                               padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size,
        num_workers,
        shuffle=True,
        pin_memory=True
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader