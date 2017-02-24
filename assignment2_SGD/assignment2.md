

```python
import tensorflow as tf
```


```python
from __future__ import print_function
```


```python
import numpy as np
```


```python
from six.moves import cPickle as pickle
from six.moves import range
```


```python
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['trainDataset']
    train_labels = save['trainLabels']
    valid_dataset = save['validDataset']
    valid_labels = save['validLabels']
    test_dataset = save['testDataset']
    test_labels = save['testLabels']
    
    #free memory
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)



```python
image_size = 28
num_labels = 10
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)



```python
# With gradient descent training, even this much data is prohibitive
# Subset the training data fro faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph
    tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
    tf_train_labels = tf.constant(train_labels[:train_subset,:])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can ben optimized). We take the average of this 
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_train_labels, logits = logits))
    
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
```


```python
num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
           / predictions.shape[0])

with tf.Session(graph = graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the 
    # biases.
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' %(step, l))
            print('Training accuracy: %.lf%%' % accuracy(predictions, train_labels[:train_subset]))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation accuracy: %.lf%%' %accuracy(
                valid_prediction.eval(), valid_labels))
    print ('Test accuracy: %.lf%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Loss at step 0: 16.871367
    Training accuracy: 7%
    Validation accuracy: 10%
    Loss at step 100: 2.304250
    Training accuracy: 72%
    Validation accuracy: 71%
    Loss at step 200: 1.841388
    Training accuracy: 75%
    Validation accuracy: 73%
    Loss at step 300: 1.603167
    Training accuracy: 76%
    Validation accuracy: 74%
    Loss at step 400: 1.446740
    Training accuracy: 77%
    Validation accuracy: 74%
    Loss at step 500: 1.330995
    Training accuracy: 78%
    Validation accuracy: 75%
    Loss at step 600: 1.239847
    Training accuracy: 78%
    Validation accuracy: 75%
    Loss at step 700: 1.165227
    Training accuracy: 79%
    Validation accuracy: 75%
    Loss at step 800: 1.102592
    Training accuracy: 79%
    Validation accuracy: 75%
    Test accuracy: 83%



```python
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will
    # be fed at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    
```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step%500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.lf%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.lf%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.lf%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 17.394146
    Minibatch accuracy: 9%
    Validation accuracy: 13%
    Minibatch loss at step 500: 2.054562
    Minibatch accuracy: 71%
    Validation accuracy: 75%
    Minibatch loss at step 1000: 0.949314
    Minibatch accuracy: 78%
    Validation accuracy: 76%
    Minibatch loss at step 1500: 1.217761
    Minibatch accuracy: 75%
    Validation accuracy: 77%
    Minibatch loss at step 2000: 1.192627
    Minibatch accuracy: 79%
    Validation accuracy: 78%
    Minibatch loss at step 2500: 1.121921
    Minibatch accuracy: 78%
    Validation accuracy: 78%
    Minibatch loss at step 3000: 1.259130
    Minibatch accuracy: 73%
    Validation accuracy: 78%
    Test accuracy: 86%



```python
# Add one layer RELU

# Weight initial function
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

# Bias initial function
def bias_variable(shape):
    return tf.Variable(0.1, shape)

graph = tf.Graph()
with graph.as_default():
    
    # Input data. The reason to use placeholder is that: we use minibatch
    tf_train_dataset = tf.placeholder(tf.float32,
                                     shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # First layer neural network
    
    hidden_size_layer1 = 1024
    
    hidden_weights = weight_variable([image_size * image_size, hidden_size_layer1])
    hidden_biases = bias_variable([hidden_size_layer1])
    hidden_layer1 = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights)+hidden_biases)
    
    output_weights = weight_variable([hidden_size_layer1, num_labels])
    output_biases = bias_variable([num_labels])
    
    # Training computation.
    logits = tf.matmul(hidden_layer1, output_weights) + output_biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(
            tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights)+hidden_biases),
            output_weights) + output_biases)
    test_prediction = tf.nn.softmax(
        tf.matmul(
            tf.nn.relu(
                tf.matmul(tf_test_dataset, hidden_weights)+hidden_biases)
            , output_weights) + output_biases)
```


```python
# Run.
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size),:]
        batch_labels = train_labels[offset:(offset + batch_size),:]
        
        feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if(step % 500) == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3.973919
    Minibatch accuracy: 12.5%
    Validation accuracy: 29.6%
    Test accuracy: 32.4%
    Minibatch loss at step 500: 0.734187
    Minibatch accuracy: 76.6%
    Validation accuracy: 84.5%
    Test accuracy: 91.4%
    Minibatch loss at step 1000: 0.280464
    Minibatch accuracy: 92.2%
    Validation accuracy: 86.5%
    Test accuracy: 93.0%
    Minibatch loss at step 1500: 0.426657
    Minibatch accuracy: 85.9%
    Validation accuracy: 87.3%
    Test accuracy: 93.8%
    Minibatch loss at step 2000: 0.340907
    Minibatch accuracy: 91.4%
    Validation accuracy: 87.8%
    Test accuracy: 94.1%
    Minibatch loss at step 2500: 0.415300
    Minibatch accuracy: 88.3%
    Validation accuracy: 88.3%
    Test accuracy: 94.3%
    Minibatch loss at step 3000: 0.447304
    Minibatch accuracy: 86.7%
    Validation accuracy: 88.3%
    Test accuracy: 94.4%



```python

```
