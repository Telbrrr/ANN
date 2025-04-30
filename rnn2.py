import numpy as np

#test sentence
vocab = ['visca', 'barca', 'catalunya', 'campeons']
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)

# Training data: wanted word catalunya
training_data = [
    (['visca', 'barca', 'visca'], 'catalunya'),
    (['visca', 'barca', 'catalunya'], 'campeons'),
    (['barca', 'visca', 'barca'], 'visca'),
    (['visca', 'visca', 'visca'], 'barca'),
]

# One-hot encoder
def one_hot(index, size):
    vec = np.zeros((size, 1))
    vec[index] = 1
    return vec

# Model parameters
hidden_size = 8
learning_rate = 0.1

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

# Training loop
for epoch in range(1000):
    total_loss = 0

    for seq, target_word in training_data:
        xs = [one_hot(word_to_ix[w], vocab_size) for w in seq]
        target_index = word_to_ix[target_word]
        hs = {-1: np.zeros((hidden_size, 1))}

        # Forward pass
        for t in range(len(xs)):
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)

        y = np.dot(Why, hs[len(xs)-1]) + by
        probs = np.exp(y) / np.sum(np.exp(y))
        loss = -np.log(probs[target_index])
        total_loss += loss

        # Backward pass
        dy = probs
        dy[target_index] -= 1
        dWhy = np.dot(dy, hs[len(xs)-1].T)
        dby = dy
        dh = np.dot(Why.T, dy)

        for t in reversed(range(len(xs))):
            dtanh = (1 - hs[t] ** 2) * dh
            dWxh = np.dot(dtanh, xs[t].T)
            dWhh = np.dot(dtanh, hs[t-1].T)
            dbh = dtanh
            dh = np.dot(Whh.T, dtanh)

            # Update weights
            Wxh -= learning_rate * dWxh
            Whh -= learning_rate * dWhh
            Why -= learning_rate * dWhy
            bh -= learning_rate * dbh
            by -= learning_rate * dby

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0]:.4f}")

# Test on original: ['visca', 'barca', 'visca']
test_seq = ['visca', 'barca', 'visca']
xs = [one_hot(word_to_ix[w], vocab_size) for w in test_seq]
hs = {-1: np.zeros((hidden_size, 1))}

print(" Hidden state progression:")
for t in range(len(xs)):
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    print(f"Step {t+1} - Hidden: {hs[t].flatten()}")

y = np.dot(Why, hs[len(xs)-1]) + by
probs = np.exp(y) / np.sum(np.exp(y))
predicted = ix_to_word[np.argmax(probs)]

print(" Final Prediction for 'visca barca visca':", predicted)