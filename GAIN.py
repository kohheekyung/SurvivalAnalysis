import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf


# data loc
data_name = './data/2020010629.xlsx'

# Training parmeter
miss_rate = 0.2
batch_size = 128
hint_rate = 0.9
alpha = 100
epoch = 10000

# Imputed data properties
imputed_columns = list()
imputed_RID = list()


def binary_sampler(p, rows, cols):
    # random matrix
    # random mas
    random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (random_matrix < p)
    return binary_random_matrix

def data_loader(file_name, miss_rate):
    # load data

    data_x = pd.read_excel(file_name, sheet_name="Sheet1")
    data_y = data_x[['CONVERTER', 'CONV_TIME']]

    # Subject ID
    imputed_RID = data_x['RID']

    # Remove unnecessary columns (features), remove first 9 columns
    remove_columns = list(data_x.columns)[0:7]
    remove_columns.append('FLDSTRENG')
    remove_columns.append('FSVERSION')
    remove_columns.append('IMAGEUID')
    remove_columns.append('EXAMDATE_bl')
    remove_columns.append('FLDSTRENG_bl')
    remove_columns.append('FSVERSION_bl')
    # remove_columns.append('ABETA')
    remove_columns.extend(list(data_x.columns)[-7:])

    print('Removing columns:', remove_columns)
    data_x = data_x.drop(remove_columns, axis=1)

    no, dim = data_x.shape
    print(no, dim)

    # Change string values to int label
    string_vars = ['DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'DX']
    # for each categorical var, convert to int label
    '''
    Convert DX_bl
    - CN 0 EMCI  1 LMCI 2 SMC 3

    Convert PTGENDER
    - Female 0  Male 1

    Convert PTETHCAT
    - Hisp/Latino 0 Not Hisp/Latino 1

    Convert PTRACCAT
    - Asian 0 Black 1 More than one 2 White 3

    Convert PTMARRY
    - Divorced 0  Married 1 Never married 2 Widowed 3
    Converting DX
    - CN 0 MCI 1
   '''
    for var in string_vars:
        print('Converting', var, 'to  label')

        one_hot_df = pd.get_dummies(data_x[var])
        print(one_hot_df.columns)

        for i in range(len(one_hot_df.columns)):
            label = one_hot_df.columns[i]

            data_x[var] = data_x[var].apply(lambda x: i if x == label else x)
            print(label, i)

    # Chane sting value to int
    # clip values range of [ 200,1700]
    data_x['ABETA_bl'] = data_x['ABETA_bl'].apply(lambda x: '1700' if x == '>1700' else x)
    data_x['ABETA_bl'] = data_x['ABETA_bl'].apply(lambda x: '200' if x == '<200' else x)
    data_x['ABETA'] = data_x['ABETA'].apply(lambda x: '1700' if x == '>1700' else x)
    data_x['ABETA'] = data_x['ABETA'].apply(lambda x: '200' if x == '<200' else x)

    print(data_x.head())

    # This will be dataset colums of imputed data
    imputed_columns.extend(data_x.columns.values.tolist())
    print('imputed', imputed_columns)

    # masked dataset = hint for gain
    data_m = binary_sampler(1 - miss_rate, no, dim)

    # dataset with missed value (= 0)
    miss_data_x = data_x.copy()
    #miss_data_x[data_x == None] = np.nan

    return np.array(data_x, dtype=np.float64), np.array(miss_data_x, dtype=np.float64), np.array(data_m,
                                                                                                 dtype=np.float64), data_y, imputed_RID

# nomalize data in range of [0, 1]
def normalization(data, parameters=None):

    _, dim = data.shape
    norm_data = data.copy()

    min = np.zeros(dim)
    max = np.zeros(dim)

    # for each feature
    for i in range(dim):
        min[i] = np.nanmin(norm_data[:, i])
        norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
        max[i] = np.nanmax(norm_data[:, i])
        norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)
    #  save min / max val of each feature
    norm_parameters = {'min_val': min,
                       'max_val': max}
    return norm_data, norm_parameters


# renorm [0, 1] to original feature range
def renormalization(norm_data, norm_parameters):
    min = norm_parameters['min_val']
    max = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    # each feature
    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min[i]
    return renorm_data

# random vector x for gain
def rand_vec(size):

    in_dim = size[0]
    stddev = 1. / tf.sqrt(in_dim / 2.)

    return tf.random_normal(shape=size, stddev=stddev)

# minibatch index
def sample_batch_index(total, batch_size):

    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

# generated uniform random matrix rang [low , high ]
def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=[rows, cols])


# Rounding categorical data to be labels
def rounding(imputed_data, data_x):
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]

        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data

# Model evaluation method
def rmse_loss(ori_data, imputed_data, data_m):
    # root meen square error
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # only for missing values
    ori_data[np.isnan(ori_data)] = 0
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse

## GAIN functions
# Generator
def build_generator(x, m, G_W1, G_W2, G_W3, G_b1, G_b2, G_b3):
    # Concatenate Mask and Data
    inputs = tf.concat(values=[x, m], axis=1)

    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob

# Discriminator
def build_discriminator(x, h, D_W1, D_W2, D_W3, D_b1, D_b2, D_b3):
    # Concatenate Data and Hint
    inputs = tf.concat(values=[x, h], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

def train(data_x):

    # mask data
    data_m = 1 - pd.isnull(data_x)
    # data & hint_m shape
    no, dim = data_x.shape
    h_dim = int(dim)

    # Input (with random vec)
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator Weight / Bias with input_dim
    D_W1 = tf.Variable(rand_vec([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(rand_vec([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(rand_vec([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))

    # Generator Weight / Bias with input_dim
    G_W1 = tf.Variable(rand_vec([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(rand_vec([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3 = tf.Variable(rand_vec([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    ## build generator
    Generator = build_generator(X, M , G_W1, G_W2, G_W3, G_b1, G_b2, G_b3)

    ## build discriminator
    Hat_X = X * M +  Generator  * (1 - M)
    Discriminator = build_discriminator(Hat_X, H,D_W1, D_W2, D_W3, D_b1, D_b2, D_b3)

    ## loss
    D_loss = -tf.reduce_mean(M * tf.log(  Discriminator + 1e-8)  + (1 - M) * tf.log(1. -   Discriminator+ 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M *  Generator ) ** 2) / tf.reduce_mean(M)
    G_loss =  -tf.reduce_mean((1 - M) * tf.log(  Discriminator + 1e-8)) + alpha * MSE_loss

    ## Adam
    D_opt = tf.train.AdamOptimizer().minimize(D_loss, var_list=[D_W1, D_W2, D_W3, D_b1, D_b2, D_b3])
    G_opt = tf.train.AdamOptimizer().minimize(G_loss, var_list=[G_W1, G_W2, G_W3, G_b1, G_b2, G_b3])

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    # Start train
    for it in tqdm(range(epoch)):

        # sampling data batch
        batch_idx = sample_batch_index(no, batch_size)

        # get batch
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]

        # sample random vec
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

        # Sample hint vec
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # random vec & unmasked vec
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_opt, D_loss],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_opt,  -tf.reduce_mean((1 - M) * tf.log(  Discriminator + 1e-8)) , MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([Generator], feed_dict={X: X_mb, M: M_mb})[0]
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)
    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data


# Load data and introduce missingness
ori_data_x, miss_data_x, data_m, data_y, imputed_RID  = data_loader(data_name, miss_rate)
# print(ori_data_x)
# print(miss_data_x)
# print(data_m)

# Impute missing data using gain
imputed_data_x = train(miss_data_x)

# Report the RMSE performance
rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
print('RMSE Performance: ' + str(np.round(rmse, 4)))

# Save new imputed data
imputed_data_df = pd.DataFrame(imputed_data_x, columns = imputed_columns)
imputed_data_df[['CONVERTER','CONV_TIME']] = data_y
imputed_data_df['RID'] = imputed_RID
imputed_data_df.head()
imputed_data_df.to_csv('./data/2020010629_Imputed.csv', index = False)
