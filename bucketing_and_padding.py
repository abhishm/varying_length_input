import numpy as np

class SimpleIterator(object):
    def __init__(self, df, batch_size):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size

    def __iter__(self):
        index = 0
        while index + self.batch_size -1 < len(df):
            batch = self.df.ix[index:index + self.batch_size]
            input_ = batch["as_numbers"]
            target = batch["gender"] * 3 + batch["age_bracket"]
            length = batch["length"]
            yield input_, target, length
            index += self.batch_size

class PaddedIterator(object):
    def __init__(self, df, batch_size):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.index = 0
    def __iter__(self):
        while self.index + self.batch_size < len(self.df):
            yield self.padding(self.df.ix[self.index:self.index + self.batch_size - 1])
            self.index += self.batch_size

    def padding(self, batch):
        batch_length = batch["length"]
        max_length = np.max(batch_length)
        inputs = batch["as_numbers"]
        padded_inputs = np.zeros((batch.shape[0], max_length), dtype=int)
        for i, (input_, length)  in enumerate(zip(inputs, batch_length)):
            padded_inputs[i, :length] = input_
        target = batch["gender"] * 3 + batch["age_bracket"]

        return padded_inputs, target.values, batch_length.values

class PaddedBucketedIterator(object):
    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.bucket_boundaries = [4, 11, 14, 18, 23, 30]
        self.dfs = []
        for i in range(len(self.bucket_boundaries) - 1):
            self.dfs.append((self.df[(self.df.length > self.bucket_boundaries[i]) &
                                    (self.df.length <= self.bucket_boundaries[i + 1])]
                             .reset_index()))
        self.num_buckets = len(self.dfs)
        self.indexes = [0] * self.num_buckets

    def __iter__(self):
        while self.dfs:
            bucket_number = np.random.choice(len(self.dfs))
            df = self.dfs[bucket_number]
            index = self.indexes[bucket_number]
            if index + self.batch_size < len(df):
                yield self.padding(df.ix[index:index + self.batch_size - 1])
                self.indexes[bucket_number] += self.batch_size
            else:
                self.dfs.pop(bucket_number)

    def padding(self, batch):
        batch_length = batch["length"]
        max_length = np.max(batch_length)
        inputs = batch["as_numbers"]
        padded_inputs = np.zeros((batch.shape[0], max_length), dtype=int)
        for i, (input_, length)  in enumerate(zip(inputs, batch_length)):
            padded_inputs[i, :length] = input_
        target = batch["gender"] * 3 + batch["age_bracket"]

        return padded_inputs, target.values, batch_length.values
