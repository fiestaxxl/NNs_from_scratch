import numpy as np
from collections import defaultdict
from torch.utils import data
import matplotlib.pyplot as plt

def generate_dataset(num_sequences=100):
    """
    Generates a number of sequences as our dataset.

    Args:
     `num_sequences`: the number of sequences to be generated.

    Returns a list of sequences.
    """
    samples = []

    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['c'] * num_tokens + ['e'] * num_tokens + ['EOS']
        samples.append(sample)

    return samples


def sequences_to_dicts(sequences):
    """
    Creates word_to_idx and idx_to_word dictionaries for a list of sequences.
    """
    # A bit of Python-magic to flatten a nested list
    flatten = lambda l: [item for sublist in l for item in sublist]

    # Flatten the dataset
    all_words = flatten(sequences)

    # Count number of word occurences
    word_count = defaultdict(int)
    for word in flatten(sequences):
        word_count[word] += 1

    # Sort by frequency
    word_count = sorted(list(word_count.items()), key=lambda l: -l[1])

    # Create a list of all unique words
    unique_words = [item[0] for item in word_count]

    # Add UNK token to list of words
    unique_words.append('UNK')

    # Count number of sequences and number of unique words
    num_sentences, vocab_size = len(sequences), len(unique_words)

    # Create dictionaries so that we can go from word to index and back
    # If a word is not in our vocabulary, we assign it to token 'UNK'
    word_to_idx = defaultdict(lambda: num_words)
    idx_to_word = defaultdict(lambda: 'UNK')

    # Fill dictionaries
    for idx, word in enumerate(unique_words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sentences, vocab_size


class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def create_datasets(sequences, dataset_class=Dataset, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []

        # Append inputs and targets s.t. both lists contain L-1 words of a sentence of length L
        # but targets are shifted right by one so that we can predict the next word
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])

        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)

    # Create datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set


def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.

    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary

    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)

    # Set the appropriate element to one
    one_hot[idx] = 1

    return one_hot


def one_hot_encode_sequence(sequence, vocab_size, word_to_idx):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.

    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary

    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence], dtype=np.int32)

    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)

    return encoding


def train(rnn, num_epochs, validation_set, training_set, test_set, idx_to_word, word_to_idx, vocab_size, lr):
    training_loss, validation_loss = [], []
    hidden_state = np.zeros((rnn.hidden_size,1))
    # For each epoch
    for i in range(num_epochs):

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # For each sentence in validation set
        for inputs, targets in validation_set:

            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            targets_one_hot = one_hot_encode_sequence(targets, vocab_size, word_to_idx)

            # Forward pass
            #outputs, hidden_states = rnn(inputs_one_hot, hidden_state)
            outputs, hidden_states = rnn(inputs_one_hot)

            # Backward pass

            #loss = compute_cce(targets_one_hot,outputs)
            loss = rnn.backward(outputs,targets_one_hot, do_grad=False)


            # Update loss
            epoch_validation_loss += loss

        # For each sentence in training set
        for inputs, targets in training_set:

            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
            targets_one_hot = one_hot_encode_sequence(targets, vocab_size, word_to_idx)

            # Re-initialize hidden state
            hidden_state = np.zeros_like(hidden_state)

            # Forward pass
            outputs, hidden_states = rnn(inputs_one_hot, hidden_state)
            # Backward pass
            loss = rnn.backward(outputs,targets_one_hot, do_grad=True)
            rnn.step(lr)


            # Update loss
            epoch_training_loss += loss

        # Save loss for plot
        training_loss.append(epoch_training_loss/len(training_set))
        validation_loss.append(epoch_validation_loss/len(validation_set))

        # Print loss every 100 epochs
        if i % 25 == 0:
            print(f'\nEpoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')


    # Get first sentence in test set
    inputs, targets = test_set[1]

    # One-hot encode input and target sequence
    inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
    targets_one_hot = one_hot_encode_sequence(targets, vocab_size, word_to_idx)

    # Initialize hidden state as zeros
    hidden_state = np.zeros((rnn.hidden_size,1))

    # Forward pass
    outputs, hidden_states = rnn(inputs_one_hot)
    output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
    print('Input sentence:')
    print(inputs)

    print('\nTarget sequence:')
    print(targets)

    print('\nPredicted sequence:')
    print([idx_to_word[np.argmax(output)] for output in outputs])

    # Plot training and validation loss
    epoch = np.arange(len(training_loss))
    plt.figure()
    plt.plot(epoch, training_loss, 'r', label='Training loss',)
    plt.plot(epoch, validation_loss, 'b', label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch'), plt.ylabel('NLL')
    plt.show()