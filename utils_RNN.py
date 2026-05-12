import numpy as np
from collections import Counter
from urllib.parse import urlparse
import re

# 1. Basic URL loading
def load_dataset(path):
    urls, labels = [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            # Must have at least label + URL
            if len(parts) < 2:
                print(f"[WARN] Skipping malformed line {i}: {line[:100]}")
                continue

            label_str = parts[0].strip()

            # Only accept valid labels (-1 or 1)
            if label_str not in {"-1", "1"}:
                print(f"[WARN] Invalid label at line {i}: {label_str}")
                continue

            url = parts[1].strip()

            if not url:
                print(f"[WARN] Empty URL at line {i}")
                continue

            labels.append(int(label_str))
            urls.append(url)

    print(f"[INFO] Loaded {len(urls)} valid samples")

    return urls, np.array(labels)


# 2. Tokenization
def tokenize_url(url, delimit_mode=1):
    url = url.lower()
    tokens = []
    buffer = ""

    for ch in url:
        if ch.isalnum():
            buffer += ch
        else:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            if delimit_mode == 1:
                tokens.append(ch)

    if buffer:
        tokens.append(buffer)

    return tokens


# 3. Character n-grams inside words
def char_ngrams(word, n=3):
    # Generate character n-grams with boundary markers. Example: login -> <lo, log, ogi, gin, in>
    word = f"<{word}>"
    return [word[i:i+n] for i in range(len(word) - n + 1)]


def build_ngram_vocab(data, n=3, min_freq=1):
    # Build ngram vocabulary. 'data' can be a list of tokenized URLs (list of list of strings) or a list of words (list of strings).
    counter = Counter()

    for item in data:
        if isinstance(item, list):
            # case: list of tokenized URLs
            for word in item:
                counter.update(char_ngrams(word, n))
        else:
            # case: list of words
            counter.update(char_ngrams(item, n))

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for ng, freq in counter.items():
        if freq >= min_freq:
            vocab[ng] = len(vocab)

    print(f"[INFO] Built ngram vocab")
    return vocab


def encode_word_ngrams_sample(tokenized_url, vocab, max_words, max_subwords, n=3):
    # Encode a single URL into ngram IDs. Returns shape: [max_words, max_subwords]
    def char_ngrams(word):
        word = f"<{word}>"
        return [word[i:i+n] for i in range(len(word) - n + 1)]

    encoded_url = []

    for word in tokenized_url[:max_words]:
        ngrams = char_ngrams(word)[:max_subwords]
        ids = [vocab.get(ng, 0) for ng in ngrams]
        ids += [0] * (max_subwords - len(ids))
        encoded_url.append(ids)

    # pad words
    while len(encoded_url) < max_words:
        encoded_url.append([0] * max_subwords)

    return np.array(encoded_url, dtype=np.int32)


# 4. Word-level vocabulary
def build_word_vocab(tokenized_urls, min_freq=1):
    counter = Counter()

    for url in tokenized_urls:
        counter.update(url)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"[INFO] Built word-level vocabulary")
    return vocab


def encode_words(tokenized_urls, vocab, max_words):
    output = []

    for url in tokenized_urls:
        ids = [vocab.get(w, 1) for w in url[:max_words]]
        ids += [0] * (max_words - len(ids))
        output.append(ids)

    print(f"[INFO] Encoded words")
    return np.array(output, dtype=np.int32)


# 5. Character-level encoding (full URL)
def build_char_vocab(urls):
    chars = set("".join(urls))
    vocab = {"<PAD>": 0}

    for c in sorted(list(chars)):
        vocab[c] = len(vocab)

    print(f"[INFO] Built char vocab")
    return vocab


def encode_chars(urls, vocab, max_len):
    output = []

    for url in urls:
        seq = [vocab.get(c, 0) for c in url[:max_len]]
        seq += [0] * (max_len - len(seq))
        output.append(seq)

    print(f"[INFO] Encoded characters")
    return np.array(output, dtype=np.int32)


# 6. Structured Tokenization
def tokenize_url_structure(url):
    # Parses a URL into structured tokens: scheme, domain parts, port, path parts, query parts
    if not url.startswith(('http://', 'https://', 'ftp://')): # just in case normalization missed one
        url_to_parse = 'http://' + url
    else:
        url_to_parse = url
        
    parsed = urlparse(url_to_parse.lower())
    tokens = []

    if parsed.scheme:
        tokens.append(parsed.scheme)

    netloc = parsed.netloc.split(':')
    host_parts = netloc[0].split('.')
    tokens.extend([p for p in host_parts if p])

    if len(netloc) > 1:
        tokens.append(netloc[1]) # port

    path_parts = [p for p in parsed.path.split('/') if p]
    tokens.extend(path_parts)

    if parsed.query:
        query_parts = re.split(r'[=&]', parsed.query)
        tokens.extend([q for q in query_parts if q])

    return tokens

def build_struct_vocab(tokenized_structs, min_freq=1):
    counter = Counter()
    for t in tokenized_structs:
        counter.update(t)
        
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, freq in counter.items():
        if freq >= min_freq:
            vocab[w] = len(vocab)
            
    print(f"[INFO] Built struct vocabulary")
    return vocab

def encode_structs(tokenized_structs, vocab, max_len):
    output = []
    for struct in tokenized_structs:
        ids = [vocab.get(w, 1) for w in struct[:max_len]]
        ids += [0] * (max_len - len(ids))
        output.append(ids)
    print(f"[INFO] Encoded structured tokens")
    return np.array(output, dtype=np.int32)


# 7. Train/test split
def split_dataset(labels, dev_ratio=0.1, seed=42):
    np.random.seed(seed)

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    np.random.shuffle(pos)
    np.random.shuffle(neg)

    def split(arr):
        cut = int(len(arr) * (1 - dev_ratio))
        return arr[:cut], arr[cut:]

    pos_train, pos_test = split(pos)
    neg_train, neg_test = split(neg)

    train_idx = np.concatenate([pos_train, neg_train])
    test_idx = np.concatenate([pos_test, neg_test])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    print(f"[INFO] Train/test split done")
    return train_idx, test_idx


# 8. Helpers
def one_hot(labels):
    # -1 -> 0, +1 -> 1 (only for tensor representation)
    idx = (labels == 1).astype(np.int32)

    out = np.zeros((len(labels), 2), dtype=np.float32)
    out[np.arange(len(labels)), idx] = 1.0
    return out


def batch_iter(x, y, batch_size, shuffle=True):
    idx = np.arange(len(y))

    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]

        # CASE 1: single input tensor
        if isinstance(x, np.ndarray):
            yield x[batch_idx], y[batch_idx]

        # CASE 2: tuple of tensors (multi-input model)
        else:
            batch_x = tuple(xi[batch_idx] for xi in x)
            yield batch_x, y[batch_idx]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# 9. Output formatting
def save_predictions(labels, preds, scores, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("label\tpredict\tscore\n")

        for l, p, s in zip(labels, preds, scores):
            l_out = 1 if l == 1 else -1
            p_out = 1 if p == 1 else -1
            score = softmax(s)[1]

            f.write(f"{l_out}\t{p_out}\t{score}\n")