#generating test and training data
import numpy as np



data = np.loadtxt('data.txt')


NUM_ROWS = data.shape[0]

training_length = np.round(NUM_ROWS * .8)


indices = np.random.choice(NUM_ROWS, training_length, replace=False)

training_data = data[indices, :]


test_data = np.delete(data, indices, 0)

np.savetxt('test.txt', test_data, fmt='%s')
np.savetxt('train.txt', training_data, fmt='%s')

f = open('train_indices.txt', 'w')
for i in indices:
    f.write('{0}\n'.format(i))

f.close()

test_indices = np.delete(np.arange(NUM_ROWS), indices)
f = open('test_indices.txt', 'w')

for i in test_indices:
    f.write('{0}\n'.format(i))
f.close()
