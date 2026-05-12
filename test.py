import argparse
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from TextCNN import URLNetModel
from utils import (
    load_dataset,
    tokenize_url,
    encode_words
)

tf.disable_v2_behavior()


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--char_dict", required=True)
    parser.add_argument("--word_dict", required=True)
    parser.add_argument("--ngram_dict", required=True)

    parser.add_argument("--max_words", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=200)
    parser.add_argument("--max_subwords", type=int, default=20)

    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--filter_sizes", default="3,4,5,6")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_file", default="results.txt")

    return parser.parse_args()


# Load vocabs
def load_resources(args):
    urls, labels = load_dataset(args.data_path)

    char_vocab = pickle.load(open(args.char_dict, "rb"))
    word_vocab = pickle.load(open(args.word_dict, "rb"))
    ngram_vocab = pickle.load(open(args.ngram_dict, "rb"))

    tokenized = [tokenize_url(u) for u in urls]

    return urls, tokenized, labels, char_vocab, word_vocab, ngram_vocab


# Char encoding
def encode_chars_fixed(urls, char_vocab, max_len):
    PAD = 0

    batch = []
    for url in urls:
        # enforce character length constraint
        url = url[:max_len]

        encoded = [char_vocab.get(c, 0) for c in url]

        # pad with <PAD>
        if len(encoded) < max_len:
            encoded += [PAD] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]

        batch.append(encoded)

    return np.array(batch, dtype=np.int32)


# ngram encoding
def encode_ngrams_fixed(tokenized_batch, ngram_vocab, max_words, max_subwords):
    PAD = 0
    batch_size = len(tokenized_batch)

    # Preallocate full tensor
    batch = np.zeros((batch_size, max_words, max_subwords), dtype=np.int32)

    for b_idx, tokens in enumerate(tokenized_batch):
        for w_idx in range(min(len(tokens), max_words)):
            w = tokens[w_idx]
            ngrams = []
            ngrams_append = ngrams.append  # local binding

            L = len(w)

            # Early exit if word empty
            if L == 0:
                continue

            # Generate substrings but stop when max_subwords reached
            for i in range(L):
                # bound j to avoid useless work
                max_j = min(i + max_subwords, L)
                for j in range(i + 1, max_j + 1):
                    gram = w[i:j]
                    idx = ngram_vocab.get(gram)
                    if idx is not None:
                        ngrams_append(idx)
                        if len(ngrams) == max_subwords:
                            break
                if len(ngrams) == max_subwords:
                    break

            # Write directly into preallocated array
            if ngrams:
                batch[b_idx, w_idx, :len(ngrams)] = ngrams

    return batch


# Stream batches
def stream_batches(urls, tokenized, labels,
                   char_vocab, word_vocab, ngram_vocab,
                   args):

    n = len(labels)

    for i in range(0, n, args.batch_size):

        batch_urls = urls[i:i + args.batch_size]
        batch_tok = tokenized[i:i + args.batch_size]
        batch_y = labels[i:i + args.batch_size]

        # char (fixed 200 length)
        char_data = encode_chars_fixed(
            batch_urls,
            char_vocab,
            args.max_chars
        )

        # word
        word_data = encode_words(
            batch_tok,
            word_vocab,
            args.max_words
        )

        # ngrams
        if args.mode in (4, 5):
            ngram_data = encode_ngrams_fixed(
                batch_tok,
                ngram_vocab,
                args.max_words,
                args.max_subwords
            )
        else:
            ngram_data = None

        # Mode mapping
        if args.mode == 1:
            x = char_data
        elif args.mode == 2:
            x = word_data
        elif args.mode in (3, 4, 5):
            x = (char_data, word_data, ngram_data) if args.mode == 5 else (char_data, word_data)
            # Dummy logic for ngram mappings inside stream_batches if mode=4,5 
            # since they are not used but I don't know how to remove them without breaking everything

        yield x, batch_y


# Feed dict
def build_feed(model, batch_x, args):
    feed = {model.dropout: 1.0}

    if args.mode == 1:
        feed[model.input_char_seq] = batch_x

    elif args.mode == 2:
        feed[model.input_word] = batch_x

    elif args.mode == 3:
        feed[model.input_char_seq] = batch_x[0]
        feed[model.input_word] = batch_x[1]

    elif args.mode == 4:
        feed[model.input_char] = batch_x[0]
        feed[model.input_word] = batch_x[1]
        feed[model.input_char_mask] = np.ones_like(batch_x[0], dtype=np.float32)

    elif args.mode == 5:
        feed[model.input_char] = batch_x[0]
        feed[model.input_word] = batch_x[1]
        feed[model.input_char_mask] = np.ones_like(batch_x[0], dtype=np.float32)
        feed[model.input_char_seq] = batch_x[2]

    return feed


# Main
def main():
    args = parse_args()

    urls, tokenized, labels, char_vocab, word_vocab, ngram_vocab = load_resources(args)

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:

            model = URLNetModel(
                char_vocab_size=len(char_vocab) + 1,
                word_vocab_size=len(word_vocab) + 1,
                char_ngram_vocab_size=len(ngram_vocab) + 1,
                word_seq_len=args.max_words,
                char_seq_len=args.max_chars,
                embedding_dim=args.embedding_dim,
                filter_sizes=list(map(int, args.filter_sizes.split(","))),
                mode=args.mode
            )

            saver = tf.train.Saver()
            checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)

            saver.restore(sess, checkpoint)

            preds_all = []
            scores_all = []

            batches = stream_batches(
                urls, tokenized, labels,
                char_vocab, word_vocab, ngram_vocab,
                args
            )

            for batch_x, batch_y in tqdm(batches, desc="Testing"):

                feed = build_feed(model, batch_x, args)

                preds, scores = sess.run(
                    [model.predictions, model.scores],
                    feed_dict=feed
                )

                preds_all.extend(preds)
                scores_all.extend(scores)

    preds_all = np.array(preds_all)
    preds_all = np.where(preds_all == 1, 1, -1)

    from utils import save_predictions
    save_predictions(labels, preds_all, scores_all, args.output_file)


if __name__ == "__main__":
    main()