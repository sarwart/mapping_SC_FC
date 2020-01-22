import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr

# Network Parameters
# Number of connections at input and output
conn_dim = 2278 #(upper-triangle of Connectiivty matrix)
layer_dim = 1024


#Xavier initializer
initializer = tf.contrib.layers.xavier_initializer()
EPS = 1e-12

# Store layers weight & bias
weights = {
    'hidden1': tf.Variable(initializer([conn_dim, layer_dim])),
    'hidden2': tf.Variable(initializer([layer_dim, layer_dim])),
    'hidden3': tf.Variable(initializer([layer_dim, layer_dim])),
    'hidden4': tf.Variable(initializer([layer_dim, layer_dim])),
    'hidden5': tf.Variable(initializer([layer_dim, layer_dim])),
    'hidden6': tf.Variable(initializer([layer_dim, layer_dim])),
    'hidden7': tf.Variable(initializer([layer_dim, layer_dim])),
    'pred_out': tf.Variable(initializer([layer_dim, conn_dim])),

}
biases = {
    'hidden1': tf.Variable(initializer([layer_dim])),
    'hidden2': tf.Variable(initializer([layer_dim])),
    'hidden3': tf.Variable(initializer([layer_dim])),
    'hidden4': tf.Variable(initializer([layer_dim])),
    'hidden5': tf.Variable(initializer([layer_dim])),
    'hidden6': tf.Variable(initializer([layer_dim])),
    'hidden7': tf.Variable(initializer([layer_dim])),
    'pred_out': tf.Variable(initializer([conn_dim])),

}

# FC Predictor function
def predictor(x,a):
    #LAyer 1
    hidden_layer1 = tf.matmul(x, weights['hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['hidden1'])
    hidden_layer1 = tf.nn.dropout(hidden_layer1, a)
    hidden_layer1 = tf.nn.leaky_relu(hidden_layer1, 0.2)

    #Layer 2
    hidden_layer2 = tf.matmul(hidden_layer1, weights['hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['hidden2'])
    hidden_layer2 = tf.nn.dropout(hidden_layer2, a)
    hidden_layer2 = tf.nn.tanh(hidden_layer2)

    #Layer 3
    hidden_layer3 = tf.matmul(hidden_layer2, weights['hidden3'])
    hidden_layer3 = tf.add(hidden_layer3, biases['hidden3'])
    hidden_layer3 = tf.nn.dropout(hidden_layer3, a)
    hidden_layer3 = tf.nn.leaky_relu(hidden_layer3, 0.2)

    #Layer 4
    hidden_layer4 = tf.matmul(hidden_layer3, weights['hidden4'])
    hidden_layer4 = tf.add(hidden_layer4, biases['hidden4'])
    hidden_layer4 = tf.nn.dropout(hidden_layer4, a)
    hidden_layer4 = tf.nn.tanh(hidden_layer4)

    #Layer 5
    hidden_layer5 = tf.matmul(hidden_layer4, weights['hidden5'])
    hidden_layer5 = tf.add(hidden_layer5, biases['hidden5'])
    hidden_layer5 = tf.nn.dropout(hidden_layer5, a)
    hidden_layer5 = tf.nn.leaky_relu(hidden_layer5, 0.2)

    #Layer 6
    hidden_layer6 = tf.matmul(hidden_layer5, weights['hidden6'])
    hidden_layer6 = tf.add(hidden_layer6, biases['hidden6'])
    hidden_layer6 = tf.nn.dropout(hidden_layer6, a)
    hidden_layer6 = tf.nn.tanh(hidden_layer6)

    #Layer 7
    hidden_layer7 = tf.matmul(hidden_layer6, weights['hidden7'])
    hidden_layer7 = tf.add(hidden_layer7, biases['hidden7'])
    hidden_layer7 = tf.nn.dropout(hidden_layer7, a)
    hidden_layer7 = tf.nn.leaky_relu(hidden_layer7, 0.2)

    #Ouput Layer
    out_layer = tf.matmul(hidden_layer7, weights['pred_out'])
    out_layer = tf.add(out_layer, biases['pred_out'])
    out_layer = tf.nn.tanh(out_layer)

    return out_layer

#estimate intra-pFC for regularization
def compute_corr_loss(gen_sample,batch_size):
    intra_corr = 0
    count = 0
    for i in range(batch_size):
        start = gen_sample[i, :]
        for j in range(batch_size):
            if (j!=i):
                count +=1
                temp=gen_sample[j,:]
                corr = correlation_coefficient_loss(start, temp)
                intra_corr += corr
    return intra_corr/count

#calculate pearson correlation for tensors
def correlation_coefficient_loss(x, y):

    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm = tf.subtract(x,mx)
    ym = tf.subtract(y, my)
    r_num = tf.reduce_sum(tf.multiply(xm,ym))
    r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm)), tf.reduce_sum(tf.square(ym))))
    r = r_num / (r_den + EPS)

    return r


#calculate intra-pFC correlation for predited values
def compute_corr(sample):
    sp_corr = 0
    count = 0
    for i in range(np.shape(sample)[0]):
        start = sample[i, :]
        for j in range(np.shape(sample)[0]):
            if (j!=i):
                count +=1
                corr, corr1 = pearsonr(np.transpose(start), np.transpose(sample[j, :]))
                sp_corr += corr
    return sp_corr/count
