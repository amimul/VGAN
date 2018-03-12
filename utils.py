import tensorflow as tf
import pickle as pkl


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z


def load_vocab(vocab_file):
    """
    :param vocab_file:
    :return: a reversed dictionary, and a list that contains all the words
    """
    dic = pkl.load(open(vocab_file, 'rb'))
    rst = {idx: word for word, idx in dic.items()}
    # words = [word for word, idx in dic.items()]
    return rst#, words