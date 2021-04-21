# Hierarchical Structrued Attention for Paraphrase Indentification(Quora, Xianer etc) Task.
The detailed implementation please see the additional PPT.

### Requirements
- PyTorch 0.2
- tqdm
- matplotlib
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7

### Usage
First, run the script `./fetch_and_preprocess.sh` to download [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml) and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml), and generates dependency parses of the Xianer dataset using this Dependency Parser.

Second, go to `src` directory and run `python main_xianer.py` to train and test the xianer model, and have a look at `config.py` for command-line arguments. The predict result is in the 'data' directiry, and attention visualization result is in the 'img' directiry.

### Note
- PyTorch 0.1x don't support Biliear Network, and need to modify some built-in functions.
- Pretrained Chinese word embedding file can find in 'xxx/embedding/huge.readable' in xxx.
- Add 'xxx/anaconda2/bin' to the PATH in xxx can run our project properly.


### Acknowledgements
[Riddhiman Dasgupta](https://github.com/dasguptar/) for the [pyTorch implementation](https://github.com/dasguptar/treelstm.pytorch) of the tree-lstm.
[Kai Sheng Tai](https://github.com/kaishengtai/) for the [original LuaTorch implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.
