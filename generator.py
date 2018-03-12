from utils import *
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Generator(object):
    def __init__(self, vocab_size, condition_size, condition_num, batch_size, emb_dim,
                 emb_condition_dim, hidden_dim, z_dim,
                 sequence_length, start_token, vocab_file, condition_file,
                 word_vec=None, learning_rate=0.01, reward_gamma=0.95):
        self.vocab_size = vocab_size
        self.condition_size = condition_size
        self.condition_num = condition_num
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.emb_condition_dim = emb_condition_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.vocab = load_vocab(vocab_file)
        self.condition = load_vocab(condition_file)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        with tf.variable_scope('generator'):
            if word_vec is None:
                self.vocab_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
            else:
                self.vocab_embeddings = tf.Variable(embedding_matrix(word_vec, self.vocab))
            self.condition_embeddings = tf.Variable(self.init_matrix([self.condition_size, self.emb_condition_dim]))
            self.g_params.extend([self.vocab_embeddings, self.condition_embeddings])
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)
            self.g_vae_unit = self.create_vae_unit(self.g_params)
            self.g_prior_distribution_unit = self.create_prior_distribution_unit(self.g_params)
            self.g_posterior_distribution_unit = self.create_posterior_distribution_unit(self.g_params)

        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])  # input for generator
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])  # true output
        self.condition = tf.placeholder(tf.int32, shape=[self.batch_size, self.condition_num])
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])  # get from rollout policy and discriminator

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.vocab_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
            self.processed_condition = tf.nn.embedding_lookup(self.condition_embeddings, self.condition)  # batch_size x cond_emb_dim

        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            pos_miu, pos_logvar = self.g_posterior_distribution_unit(h_t[0], x_t)  # params for posterior dist.
            logits, prob = self.g_vae_unit(pos_miu, pos_logvar)  # prediction of vae, batch x vocab_size
            log_prob = tf.log(prob)
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)  # [batch_size]
            x_tp1 = tf.nn.embedding_lookup(self.vocab_embeddings, next_token)  # batch x emb_dim, x_t for next loop
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             prob), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.vocab_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        # supervised pretraining for generator
        lstm_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        vae_logits = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        vae_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        kl_losses = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)  # true input

        def _pretrain_recurrence(i, x_t, h_tm1, lstm_predictions, vae_logits, vae_predictions, kl_losses):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)  # output of lstm, batch x vocab_size
            pri_miu, pri_logvar = self.g_prior_distribution_unit(h_tm1[0])  # params for prior dist.
            pos_miu, pos_logvar = self.g_posterior_distribution_unit(h_t[0], x_t)  # params for posterior dist.
            logits, prob = self.g_vae_unit(pos_miu, pos_logvar)  # prediction of vae, batch x vocab_size
            lstm_predictions = lstm_predictions.write(i, tf.nn.softmax(o_t))  # possibility distribution
            vae_logits = vae_logits.write(i, logits)
            vae_predictions = vae_predictions.write(i, prob)
            kl_losses = kl_losses.write(i, gaussian_kld(pri_miu, pri_logvar, pos_miu, pos_logvar))
            x_tp1 = ta_emb_x.read(i)  # true next input
            return i + 1, x_tp1, h_t, lstm_predictions, vae_logits, vae_predictions, kl_losses

        _, _, _, self.lstm_predictions, self.vae_logits, self.vae_predictions, self.kl_losses = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.vocab_embeddings, self.start_token), self.h0,
                       lstm_predictions, vae_logits, vae_predictions, kl_losses))

        self.lstm_predictions = tf.transpose(self.lstm_predictions.stack(),
                                             perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.vae_logits = tf.transpose(self.vae_logits.stack(),
                                       perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.vae_predictions = tf.transpose(self.vae_predictions.stack(),
                                            perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # pretraining loss
        one_hot_x = tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.vocab_size,
                               1.0, 0.0)  # (batch * seqlen) * vocab_size
        self.lstm_loss = -tf.reduce_sum(
            one_hot_x * tf.log(
                tf.clip_by_value(tf.reshape(self.lstm_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)
        self.recon_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.vae_logits, [-1, self.vocab_size]),
                                                    labels=one_hot_x)
        ) / (self.sequence_length * self.batch_size)
        self.kl_loss = tf.reduce_sum(self.kl_losses.stack())

        self.pretrain_loss = self.lstm_loss + self.recon_loss + self.kl_loss

        # training updates
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        #######################################################################################################
        #  Unsupervised Training
        #######################################################################################################
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.vae_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def pretrain_step(self, sess, batch):
        """
        :param sess:
        :param batch: condition, first sentences, second sentences
        :return:
        """
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss,
                            self.lstm_loss, self.recon_loss, self.kl_loss],
                           feed_dict={self.condition: batch[0], self.x: batch[1], self.y: batch[2]})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit  # returning a function

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.vocab_size]))
        self.bo = tf.Variable(self.init_matrix([self.vocab_size]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

    def create_vae_unit(self, params):
        self.W_z2h = tf.Variable(self.init_matrix([self.z_dim, self.hidden_dim]))
        self.b_z2h = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.W_h2x = tf.Variable(self.init_matrix([self.hidden_dim, self.vocab_size]))
        self.b_h2x = tf.Variable(self.init_matrix([self.vocab_size]))

        params.extend([self.W_z2h, self.b_z2h, self.W_h2x, self.b_h2x])

        def unit(miu, logvar):
            """
            P(X|z)
            miu and sigma are parameters for normal distribution.
            generate z and then recover X from z.
            :param miu: batch_size * z_dim
            :param logvar: batch_size * z_dim
            :return: logits: batch_size * vocab_size
                    prob: batch_size * vocab_size, probability over all vocabulary
            """
            print("vae predicting.")
            # reparametrization
            z = sample_gaussian(miu, logvar)  # batch * z_dim
            h = tf.nn.relu(tf.matmul(z, self.W_z2h) + self.b_z2h)  # batch * h_dim
            logits = tf.matmul(h, self.W_h2x) + self.b_h2x  # batch * vocab_size(vocab_size)
            prob = tf.nn.softmax(logits)
            return logits, prob

        return unit

    def create_prior_distribution_unit(self, params):
        self.Wpr_miu = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]))
        self.Vpr_miu = tf.Variable(self.init_matrix([self.condition_num * self.emb_condition_dim, self.z_dim]))
        self.bpr_miu = tf.Variable(self.init_matrix([self.z_dim]))

        self.Wpr_sig = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]))
        self.Vpr_sig = tf.Variable(self.init_matrix([self.condition_num * self.emb_condition_dim, self.z_dim]))
        self.bpr_sig = tf.Variable(self.init_matrix([self.z_dim]))

        params.extend([self.Wpr_miu, self.Vpr_miu, self.bpr_miu, self.Wpr_sig, self.Vpr_sig, self.bpr_sig])

        def unit(h):
            """
            :param h: h_(t-1), batch * hidden_dim
            :return: miu, sigma_square for prior distribution
                    miu: batch * z_dim
                    sigma_sqare: batch * z_dim * z_dim
            """
            miu = tf.matmul(tf.reshape(self.processed_condition, [self.batch_size, -1]), self.Vpr_miu) + \
                  tf.matmul(h, self.Wpr_miu) + self.bpr_miu
            logvar = tf.matmul(tf.reshape(self.processed_condition, [self.batch_size, -1]), self.Vpr_sig) +\
                     tf.matmul(h, self.Wpr_sig) + self.bpr_sig
            return miu, logvar

        return unit

    def create_posterior_distribution_unit(self, params):
        self.Wpo_miu = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]))
        self.Upo_miu = tf.Variable(self.init_matrix([self.emb_dim, self.z_dim]))
        self.Vpo_miu = tf.Variable(self.init_matrix([self.condition_num * self.emb_condition_dim, self.z_dim]))
        self.bpo_miu = tf.Variable(self.init_matrix([self.z_dim]))

        self.Wpo_sig = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]))
        self.Upo_sig = tf.Variable(self.init_matrix([self.emb_dim, self.z_dim]))
        self.Vpo_sig = tf.Variable(self.init_matrix([self.condition_num * self.emb_condition_dim, self.z_dim]))
        self.bpo_sig = tf.Variable(self.init_matrix([self.z_dim]))

        params.extend([self.Wpo_miu, self.Upo_miu, self.Vpo_miu, self.bpo_miu,
                       self.Wpo_sig, self.Upo_sig, self.Vpo_sig, self.bpo_sig])

        def unit(h, x):
            """
            :param h: h_t, batch * hidden_dim
            :param x: x_t, batch * emb_dim
            :return: miu, sigma for posterior distribution
                    miu: batch * z_dim
                    sigma_sqare: batch * z_dim * z_dim
            """
            miu = tf.matmul(tf.reshape(self.processed_condition, [self.batch_size, -1]), self.Vpo_miu) + \
                tf.matmul(h, self.Wpo_miu) + tf.matmul(x, self.Upo_miu) + self.bpo_miu
            logvar = tf.matmul(tf.reshape(self.processed_condition, [self.batch_size, -1]), self.Vpo_sig) + \
                tf.matmul(h, self.Wpo_sig) + tf.matmul(x, self.Upo_sig) + self.bpo_sig
            return miu, logvar

        return unit

