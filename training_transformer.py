import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import pickle

from TextCNN_transformer import URLNetModel
from utils_transformer import (
    load_dataset,
    tokenize_url,
    build_word_vocab,
    build_ngram_vocab,
    build_char_vocab,
    tokenize_url_structure,
    build_struct_vocab,
    encode_structs
)

tf.disable_v2_behavior()


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="train.txt")

    parser.add_argument("--max_words", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=200)
    parser.add_argument("--max_subwords", type=int, default=20)
    parser.add_argument("--max_structs", type=int, default=50)
    parser.add_argument("--min_freq", type=int, default=1)

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--filter_sizes", type=str, default="3,4,5,6")
    parser.add_argument("--mode", type=int, default=3)

    parser.add_argument("--trans_dim", type=int, default=64)
    parser.add_argument("--trans_layers", type=int, default=3)
    parser.add_argument("--trans_heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0)

    parser.add_argument("--output_dir", type=str, default="runs/new_model/")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)

    return parser.parse_args()


# Padding helpers
def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))


def pad_2d_sequence(seq, max_len_1, max_len_2):
    seq = seq[:max_len_1]
    padded = [pad_sequence(s, max_len_2) for s in seq]
    while len(padded) < max_len_1:
        padded.append([0] * max_len_2)
    return padded


# Label helper
def labels_to_onehot(y):
    idx = (y == 1).astype(np.int32)
    out = np.zeros((len(y), 2), dtype=np.float32)
    out[np.arange(len(y)), idx] = 1.0
    return out


# Dataset
def build_dataset(args):
    urls, labels = load_dataset(args.data_path)

    tokenized = [tokenize_url(u) for u in urls]

    # word vocab
    word_vocab = build_word_vocab(tokenized, args.min_freq)

    # high frequency words
    high_freq_words = None
    if args.min_freq > 0:
        word_counts = {}
        for tokens in tokenized:
            for w in tokens:
                word_counts[w] = word_counts.get(w, 0) + 1

        high_freq_words = sorted([
            w for w, c in word_counts.items()
            if c >= args.min_freq
        ])

        print(f"Number of words with freq >= {args.min_freq}: {len(high_freq_words)}")

    # ngram vocab
    if high_freq_words is not None:
        ngram_vocab = build_ngram_vocab(high_freq_words, n=3, min_freq=args.min_freq)
    else:
        ngram_vocab = build_ngram_vocab(tokenized, n=3, min_freq=args.min_freq)

    # char vocab
    char_vocab = build_char_vocab(urls)

    # structured vocab
    struct_tokenized = [tokenize_url_structure(u) for u in urls]
    struct_vocab = build_struct_vocab(struct_tokenized, args.min_freq)

    # Encoding
    word_data = []
    for tokens in tokenized:
        encoded = [word_vocab.get(t, 1) for t in tokens]
        word_data.append(pad_sequence(encoded, args.max_words))
    word_data = np.array(word_data, dtype=np.int32)

    char_data = []
    for url in urls:
        encoded = [char_vocab.get(c, 0) for c in url]
        char_data.append(pad_sequence(encoded, args.max_chars))
    char_data = np.array(char_data, dtype=np.int32)

    struct_data = encode_structs(struct_tokenized, struct_vocab, args.max_structs)

    return {
        "word": word_data,
        "char": char_data,
        "struct": struct_data,
        "labels": labels,
        "tokenized": tokenized,
        "struct_tokenized": struct_tokenized,
        "dicts": {
            "word": word_vocab,
            "char": char_vocab,
            "ngram": ngram_vocab,
            "struct": struct_vocab
        }
    }


# Batch generator
def batch_generator(data, labels, args, shuffle=True):
    indices = np.arange(len(labels))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            
            batch_x = {
                "struct": data["struct"][batch_idx]
            }

            if args.mode == 1:
                batch_x["char"] = data["char"][batch_idx]
            elif args.mode == 2:
                batch_x["word"] = data["word"][batch_idx]
            elif args.mode in (3, 4, 5):
                batch_x["char"] = data["char"][batch_idx]
                batch_x["word"] = data["word"][batch_idx]
                # If mode 4/5 were fully implemented in stream_batches here it would do ngram

            y = labels[batch_idx]
            yield batch_idx, batch_x, y


# Feed dict
def build_feed(model, batch_x, batch_y, args, training=True):
    batch_y_oh = labels_to_onehot(batch_y)

    feed = {
        model.input_y: batch_y_oh,
        model.dropout: 0.5 if training else 1.0,
        model.input_struct: batch_x["struct"]
    }

    if args.mode == 1:
        feed[model.input_char_seq] = batch_x["char"]
    elif args.mode == 2:
        feed[model.input_word] = batch_x["word"]
    elif args.mode == 3:
        feed[model.input_char_seq] = batch_x["char"]
        feed[model.input_word] = batch_x["word"]
    elif args.mode in (4, 5):
        feed[model.input_char_seq] = batch_x["char"]
        feed[model.input_word] = batch_x["word"]
        # Dummy mask for 4/5 to prevent errors if not passed
        # feed[model.input_char_mask] = ...

    return feed


# Evaluation
def evaluate(sess, model, data, args, val_idx):
    total_loss, total_acc, total = 0, 0, 0

    for i in range(0, len(val_idx), args.batch_size):
        batch_idx = val_idx[i:i + args.batch_size]

        batch_x = {
            "struct": data["struct"][batch_idx]
        }
        
        if args.mode == 1:
            batch_x["char"] = data["char"][batch_idx]
        elif args.mode == 2:
            batch_x["word"] = data["word"][batch_idx]
        elif args.mode in (3, 4, 5):
            batch_x["char"] = data["char"][batch_idx]
            batch_x["word"] = data["word"][batch_idx]

        y_batch = data["labels"][batch_idx]

        feed = build_feed(model, batch_x, y_batch, args, training=False)

        loss, acc = sess.run([model.loss, model.accuracy], feed)

        bs = len(y_batch)
        total_loss += loss * bs
        total_acc += acc * bs
        total += bs

    return total_loss / total, total_acc / total


# Training
def train(args):
    data = build_dataset(args)

    os.makedirs(args.output_dir, exist_ok=True)

    for name in ["word", "char", "ngram", "struct"]:
        with open(os.path.join(args.output_dir, f"{name}_dict.pkl"), "wb") as f:
            pickle.dump(data["dicts"][name], f)

    n = len(data["labels"])
    val_size = max(1, int(n * 0.001))
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_idx = indices[:val_size]

    with tf.Graph().as_default():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:

            model = URLNetModel(
                char_vocab_size=len(data["dicts"]["char"]) + 1,
                word_vocab_size=len(data["dicts"]["word"]) + 1,
                char_ngram_vocab_size=len(data["dicts"]["ngram"]) + 1,
                struct_vocab_size=len(data["dicts"]["struct"]) + 1,
                word_seq_len=args.max_words,
                char_seq_len=args.max_chars,
                struct_seq_len=args.max_structs,
                embedding_dim=args.embedding_dim,
                filter_sizes=list(map(int, args.filter_sizes.split(","))),
                mode=args.mode,
                l2_reg=args.l2,
                trans_dim=args.trans_dim,
                trans_layers=args.trans_layers,
                trans_heads=args.trans_heads
            )

            optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)
            grads = optimizer.compute_gradients(model.loss)
            grads = [(tf.clip_by_norm(g, 5.0), v) for g, v in grads if g is not None]
            train_op = optimizer.apply_gradients(grads)

            sess.run(tf.global_variables_initializer())

            saver = tf.compat.v1.train.Saver()
            best = float("inf")

            train_gen = batch_generator(data, data["labels"], args)

            step = 0

            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch + 1}")

                for _ in tqdm(range(n // args.batch_size), desc="Training"):

                    batch_idx, x_batch, y_batch = next(train_gen)

                    feed = build_feed(model, x_batch, y_batch, args, training=True)

                    _, loss, acc = sess.run(
                        [train_op, model.loss, model.accuracy],
                        feed
                    )

                    if step % args.print_every == 0:
                        print(f"step {step} loss={loss:.4f} acc={acc:.4f}")

                    if step % args.eval_every == 0:
                        val_loss, val_acc = evaluate(sess, model, data, args, val_idx)
                        print(f"[VAL] loss={val_loss:.4f} acc={val_acc:.4f}")

                        if val_loss < best:
                            best = val_loss
                            saver.save(sess, os.path.join(args.output_dir, "model"), step)

                    step += 1


# Main
if __name__ == "__main__":
    args = parse_args()
    train(args)