import pickle
import collections

class Vocab():

    def __init__(self,filepath):


        self.counter=self.read_file_word_freq(filepath)

        self.word2index = {}
        self.index2word = {}
        self.add_word("pad")  # 必须第一个插入
        self.add_word("unk")

        for word,freq in self.counter.most_common(300000):
            self.add_word(word)

    def __len__(self):
        return len(self.word2index)

    def add_word(self,word):
        if word not in self.word2index:
            id=len(self.word2index)
            self.word2index[word]=id
            self.index2word[id]=word


    def read_file_word_freq(self, filepath):
        counter = collections.Counter()
        with open(filepath, "r") as f:
            for i,line in enumerate(f):
                line = line.strip().split(" ")
                counter.update(line)
                if i%10000==0:
                    print(i)
                # if i>40:
                #     break
        return counter


    def get_word_index(self,word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["unk"]

    def __getitem__(self, item):
        return self.get_word_index(item)


    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

# vocab=Vocab("zhwiki_seg.txt")
# print(vocab.counter.most_common(1000))
# vocab.save_vocab("vocab_ori")

# vocab=Vocab("test_corpus")
# vocab.save_vocab("vocab_test")