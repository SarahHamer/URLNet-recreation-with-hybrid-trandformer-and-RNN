import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class URLNetModel:
    def __init__(
        self,
        char_vocab_size,
        word_vocab_size,
        char_ngram_vocab_size,
        embedding_dim=32,
        word_seq_len=200,
        char_seq_len=200,
        filter_sizes=(3, 4, 5, 6),
        mode=3,
        l2_reg=0.0,
    ):
        self.mode = mode
        self.l2_reg = l2_reg
        self.filter_sizes = filter_sizes
        self.embedding_dim = embedding_dim
        self.l2_loss = tf.constant(0.0)

        self._build_placeholders()
        self._build_embeddings(
            char_vocab_size,
            word_vocab_size,
            char_ngram_vocab_size,
            embedding_dim,
        )

        word_features = self._word_branch(word_seq_len)
        char_features = self._char_branch(char_seq_len)

        combined = self._merge_features(word_features, char_features)

        self._build_classifier(combined)


    # Inputs
    def _build_placeholders(self):
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, 2])
        self.dropout = tf.compat.v1.placeholder(tf.float32)

        self.input_word = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.input_char_seq = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.input_char = tf.compat.v1.placeholder(tf.int32, [None, None, None])

        self.input_char_mask = tf.compat.v1.placeholder(
            tf.float32, [None, None, None, self.embedding_dim]
        )


    # Embeddings
    def _build_embeddings(
        self,
        char_vocab_size,
        word_vocab_size,
        char_ngram_vocab_size,
        emb_dim,
    ):
        with tf.compat.v1.variable_scope("embeddings"):
            self.char_seq_w = tf.Variable(
                tf.random.uniform([char_vocab_size, emb_dim], -1.0, 1.0)
            )
            self.word_w = tf.Variable(
                tf.random.uniform([word_vocab_size, emb_dim], -1.0, 1.0)
            )
            self.char_ngram_w = tf.Variable(
                tf.random.uniform([char_ngram_vocab_size, emb_dim], -1.0, 1.0)
            )

        self.word_emb = tf.nn.embedding_lookup(self.word_w, self.input_word)
        self.char_seq_emb = tf.nn.embedding_lookup(self.char_seq_w, self.input_char_seq)
        self.char_ngram_emb = tf.nn.embedding_lookup(self.char_ngram_w, self.input_char)

        self.char_ngram_emb = self.char_ngram_emb * self.input_char_mask
        self.char_ngram_sum = tf.reduce_sum(self.char_ngram_emb, axis=2)


    # CNN block
    def _conv_block(self, x, seq_len, scope):
        x = tf.expand_dims(x, -1)
        pooled_outputs = []

        for f in self.filter_sizes:
            with tf.compat.v1.variable_scope(f"{scope}_filter_{f}"):
                kernel = tf.Variable(
                    tf.random.truncated_normal(
                        [f, self.embedding_dim, 1, 256], stddev=0.1
                    )
                )
                bias = tf.Variable(tf.constant(0.1, shape=[256]))

                self.l2_loss += tf.nn.l2_loss(kernel)
                self.l2_loss += tf.nn.l2_loss(bias)

                conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="VALID")
                act = tf.nn.relu(tf.nn.bias_add(conv, bias))

                pooled = tf.nn.max_pool2d(
                    act,
                    ksize=[1, seq_len - f + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                )

                pooled_outputs.append(pooled)

        merged = tf.concat(pooled_outputs, axis=3)
        flattened = tf.reshape(merged, [-1, 256 * len(self.filter_sizes)])

        return tf.nn.dropout(flattened, rate=1 - self.dropout)


    # Word branch
    def _word_branch(self, seq_len):
        if self.mode not in [2, 3, 4, 5]:
            return None

        if self.mode in [4, 5]:
            combined = self.word_emb + self.char_ngram_sum
        else:
            combined = self.word_emb

        return self._conv_block(combined, seq_len, "word")


    # Char branch
    def _char_branch(self, seq_len):
        if self.mode not in [1, 3, 5]:
            return None

        return self._conv_block(self.char_seq_emb, seq_len, "char")


    # Merge
    def _merge_features(self, word_feat, char_feat):
        if self.mode in [3, 5]:
            with tf.compat.v1.variable_scope("word_char_concat"):
                ww = tf.compat.v1.get_variable(
                    "ww", shape=[word_feat.shape[1], 512],
                    initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                )
                bw = tf.Variable(tf.constant(0.1, shape=[512]))

                wc = tf.compat.v1.get_variable(
                    "wc", shape=[char_feat.shape[1], 512],
                    initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                )
                bc = tf.Variable(tf.constant(0.1, shape=[512]))

                self.l2_loss += tf.nn.l2_loss(ww) + tf.nn.l2_loss(bw)
                self.l2_loss += tf.nn.l2_loss(wc) + tf.nn.l2_loss(bc)

                word_proj = tf.matmul(word_feat, ww) + bw
                char_proj = tf.matmul(char_feat, wc) + bc

                return tf.concat([word_proj, char_proj], axis=1)

        elif self.mode in [2, 4]:
            return word_feat

        elif self.mode == 1:
            return char_feat


    # Classifier
    def _build_classifier(self, features):
        with tf.compat.v1.variable_scope("output"):
            w0 = tf.compat.v1.get_variable(
                "w0", shape=[features.shape[1], 512],
                initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            )
            b0 = tf.Variable(tf.constant(0.1, shape=[512]))

            w1 = tf.compat.v1.get_variable(
                "w1", shape=[512, 256],
                initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            )
            b1 = tf.Variable(tf.constant(0.1, shape=[256]))

            w2 = tf.compat.v1.get_variable(
                "w2", shape=[256, 128],
                initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            )
            b2 = tf.Variable(tf.constant(0.1, shape=[128]))

            w = tf.compat.v1.get_variable(
                "w", shape=[128, 2],
                initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            )
            b = tf.Variable(tf.constant(0.1, shape=[2]))

            # L2
            for var in [w0, b0, w1, b1, w2, b2, w, b]:
                self.l2_loss += tf.nn.l2_loss(var)

            out0 = tf.nn.relu(tf.matmul(features, w0) + b0)
            out1 = tf.nn.relu(tf.matmul(out0, w1) + b1)
            out2 = tf.nn.relu(tf.matmul(out1, w2) + b2)

            logits = tf.matmul(out2, w) + b

            self.scores = logits
            self.predictions = tf.argmax(logits, axis=1)

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.input_y
            )

            self.loss = tf.reduce_mean(loss) + self.l2_reg * self.l2_loss

            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(self.predictions, tf.argmax(self.input_y, axis=1)),
                    tf.float32,
                )
            )