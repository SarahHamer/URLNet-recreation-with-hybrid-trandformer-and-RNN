**Replicating URLNet and Enhancing It with a Hybrid Transformer and RNN Approach**
==========

Introduction
------------

This is the corresponding code for a graduate level course project from Spring 2026. The project replicates the malicious URL detection convoluted neural network (CNN) called URLNet, which has been created by Le et al. (2018) and outlined in their paper "URLNet - Learning a URL Representation with Deep
Learning for Malicious URL Detection", available here: https://arxiv.org/abs/1802.03162. URLNet was recreated and alternative versions were created that combine URLNet with a transformer and Bidirectional Long Short-Term Memory (BiLSTM) respectively. However, the baseline URLNet still outperforms both hybrid approaches during testing.

Different Modes for URLNet
------------

URLNet offers different modes, depending on which kind of URLNet is used. Character-level URLNet runs through mode 1, word-level URLNet runs through mode 2, and full URLNet (combining those two) runs through mode 3.

Example Commands to Run the Scripts
------------

Since these scripts use an outdated version of tensorflow, it is recommended to run them in a docker container or similar. The example commands also list additional parts such as 'nice' to manage memory use, which can be left out depending on the system this is run on.
<br><br>

**Baseline URLNet**

Training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python training.py --data_path processed_data_minimized/train.txt --min_freq 1 --mode 3 --embedding_dim 32 --filter_sizes 3,4,5,6 --epochs 5 --batch_size 128 --lr 0.001 --print_every 50 --eval_every 500 --output_dir runs/training_200_full
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python test.py --data_path processed_data_minimized/test.txt --char_dict runs/training_200_full/char_dict.pkl --word_dict runs/training_200_full/word_dict.pkl --ngram_dict runs/training_200_full/ngram_dict.pkl --checkpoint_dir runs/training_200_full/ --output_file test_200_full_results.txt --mode 3 --embedding_dim 32 --filter_sizes 3,4,5,6 --batch_size 128
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python auc.py --input_path ./ --input_file test_200_full_results.txt --threshold 0.5 --output_file 200_full_auc.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<br><br>
**URLNet and Transformer Hybrid**

Training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python training_transformer.py --data_path processed_data_minimized/train.txt --min_freq 1 --mode 2 --embedding_dim 32 --filter_sizes 3,4,5,6 --epochs 5 --batch_size 128 --lr 0.001 --print_every 50 --eval_every 500 --output_dir runs/training_200_word_transformer --max_structs 50 --trans_dim 64 --trans_layers 3 --trans_heads 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python test_transformer.py --data_path processed_data_minimized/test.txt --char_dict runs/training_200_word_transformer/char_dict.pkl --word_dict runs/training_200_word_transformer/word_dict.pkl --ngram_dict runs/training_200_word_transformer/ngram_dict.pkl --checkpoint_dir runs/training_200_word_transformer/ --struct_dict runs/training_200_word_transformer/struct_dict.pkl --output_file test_200_word_transformer_results.txt --mode 2 --embedding_dim 32 --filter_sizes 3,4,5,6 --batch_size 128 --max_structs 50 --trans_dim 64 --trans_layers 3 --trans_heads 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python auc.py --input_path ./ --input_file test_200_word_transformer_results.txt --threshold 0.5 --output_file transformer_200_word_auc.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<br><br>
**URLNet and RNN Hybrid**

Training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python training_RNN.py --data_path processed_data_minimized/train.txt --min_freq 1 --mode 2 --embedding_dim 32 --filter_sizes 3,4,5,6 --epochs 5 --batch_size 128 --lr 0.001 --print_every 50 --eval_every 500 --rnn_dim 64 --rnn_layers 2 --output_dir runs/training_200_word_RNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nice -n 10 python test_RNN.py --data_path processed_data_minimized/test.txt --char_dict runs/training_200_word_RNN/char_dict.pkl --word_dict runs/training_200_word_RNN/word_dict.pkl --ngram_dict runs/training_200_word_RNN/ngram_dict.pkl --struct_dict runs/training_200_word_RNN/struct_dict.pkl --checkpoint_dir runs/training_200_word_RNN/ --output_file test_200_word_RNN_results.txt --mode 2 --embedding_dim 32 --filter_sizes 3,4,5,6 --batch_size 128 --rnn_dim 64 --rnn_layers 2 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python auc.py --input_path ./ --input_file test_200_word_RNN_results.txt --threshold 0.5 --output_file RNN_200_word_auc.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets
------------

The datasets included here are a filtered version of a whole dataset with malicious and benign URLs that has been compiled specifically for this project and is available online: https://www.kaggle.com/datasets/sarahhamer/malicious-and-benign-urls-2010-2026. The versions in here are already preprocessed and formatted in a way that is compatible with the models used.
