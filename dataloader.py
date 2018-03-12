import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file, vocab_file=None, condition_file=None):
        self.first_sentences_stream, self.second_sentences_stream, self.conditions_stream = [], [], []
        with open(data_file, "r") as fin:
            data = fin.readlines()
        for poem in data:
            details = [int(t) for t in poem.split()]
            self.conditions_stream.append(details[:6])
            self.first_sentences_stream.append([3012] + details[6: 56])
            self.second_sentences_stream.append([3012] + details[56: 106])

        self.num_batch = int(len(self.conditions_stream) / self.batch_size)
        self.conditions_stream = self.conditions_stream[:self.num_batch * self.batch_size]
        self.first_sentences_stream = self.first_sentences_stream[:self.num_batch * self.batch_size]
        self.second_sentences_stream = self.second_sentences_stream[:self.num_batch * self.batch_size]
        self.condition_batch = np.split(np.array(self.conditions_stream), self.num_batch, 0)
        self.first_sentence_batch = np.split(np.array(self.first_sentences_stream), self.num_batch, 0)
        self.second_sentence_batch = np.split(np.array(self.second_sentences_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.condition_batch[self.pointer], self.first_sentence_batch[self.pointer], self.second_sentence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

