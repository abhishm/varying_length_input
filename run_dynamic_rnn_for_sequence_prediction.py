import blogs_data
import sys
from dynamic_rnn_for_sequence_prediction import DynamicRNN
from dynamic_rnn_for_seq2seq_prediction import DynamicRNNSeq2Seq
from bucketing_and_padding import PaddedIterator
from bucketing_and_padding import PaddedBucketedIterator

df = blogs_data.loadBlogs().sample(frac=1).reset_index(drop=True)
vocab, reverse_vocab = blogs_data.loadVocab()

# train-test split
train_len, test_len = int(0.8 * len(df)), int(0.2 * len(df))
train, test = df[:train_len], df[train_len:]

if sys.argv[1] == "bucketing":
    iterator =  PaddedIterator
elif sys.argv[1] == "padding":
    iterator = PaddedBucketedIterator

if sys.argv[2] == "seq2seq":
    rnn = DynamicRNNSeq2Seq
elif sys.argv[2] == "sequence":
    rnn = DynamicRNN

suffix = sys.argv[1] + "_" + sys.argv[2]
dynamic_rnn = rnn(state_size = 64,
                 batch_size = 256,
                 num_classes = 6,
                 vocab_size = len(vocab),
                 learning_rate = 0.001,
                 dropout_prob = 0.6,
                 suffix = suffix)

for epoch in range(1):
    tr_data = iter(iterator(train, 256))
    for i, batch in enumerate(tr_data):
        dynamic_rnn.update_params(batch)
