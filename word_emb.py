import torch
import torch.nn as nn
import random
from torch import optim
import time
from voc import Vocab


class SkipGram(nn.Module):
    def __init__(self, v):
        super(SkipGram, self).__init__()

        self.v = v
        self.d = 100
        self.word_emb1 = nn.Embedding(self.v, self.d)
        self.word_emb2 = nn.Embedding(self.v, self.d)

    def forward(self, x_idx_batch, context_idx_batch, other_idx_batch):
        # x_idx [batch]
        # context_idx [batch]
        # other_idx [batch][k]

        x_tensor = self.word_emb1(x_idx_batch)  # [batch][dim]
        context_tensor = self.word_emb2(context_idx_batch)  # [batch][dim]
        other_tensor = self.word_emb2(other_idx_batch)  # [batch][k][dim]

        score_context = torch.bmm(x_tensor.unsqueeze(1),context_tensor.unsqueeze(2)).squeeze(1)  # [batch]
        score_context=score_context.sigmoid().log().neg()


        score_other = torch.bmm(other_tensor, x_tensor.unsqueeze(2)).squeeze(2) #[batch][k]

        score_other=score_other.neg().sigmoid().log().sum(-1).neg() #[batch]



        loss = score_context + score_other
        loss=loss.mean()

        # loss=torch.tensor(0,dtype=float)
        # for i,x_idx in enumerate(x_idx_batch):
        #     for context_idx in context_idx_batch[i]:
        #         loss-=torch.log(torch.sigmoid(self.word_emb1(torch.tensor(x_idx)).dot(self.word_emb2(torch.tensor(context_idx)))))
        #
        #     for context_idx in other_idx_batch[i]:
        #         loss -= torch.log(torch.sigmoid(-((self.word_emb1(torch.tensor(x_idx))).dot(self.word_emb2(torch.tensor(context_idx))))))

        return loss

    def test_word_similar(self, word1, word2=None):
        idx1 = vocab.get_word_index(word1)
        if idx1==vocab["unk"]:
            print(word1,"unk")
            return
        word_emb1 = self.word_emb1(torch.tensor(idx1, device=device))

        if word2 != None:

            idx2 = vocab.get_word_index(word2)
            if idx2 == vocab["unk"]:
                print(word2, "unk")
                return
            word_emb2 = self.word_emb1(torch.tensor(idx2, device=device))
            print(word1, word2, torch.cosine_similarity(word_emb1, word_emb2, dim=0))

        word_emb1.unsqueeze_(0)
        score = torch.cosine_similarity(self.word_emb1(torch.tensor(range(0, len(vocab)), device=device)), word_emb1)
        similar_word_idx = torch.topk(score, k=200)[1]
        print(word1, [vocab.index2word[word_idx.item()] for word_idx in similar_word_idx])


class DataGenerator():
    def __init__(self,voc):
        self.voc=voc


    def sample_word_random(self):
        other_word_idx = random.randint(2,len(self.voc)-1)
        other_word=self.voc.index2word[other_word_idx]
        return other_word

    def batchIter(self):
        batch_x=[]
        batch_context=[]
        batch_other=[]

        for line_cnt,sentence in enumerate(open("zhwiki_seg.txt","r")):

            sentence_words = sentence.strip().split(" ")

            for i, word in enumerate(sentence_words):
                if vocab.get_word_index(word)==self.voc["unk"]:
                    continue

                context_words = []
                for j in range(i - m, i + m + 1):
                    if j == i or j < 0 or j >= len(sentence_words):
                        continue
                    elif vocab.get_word_index(sentence_words[j])==self.voc["unk"]:
                        continue
                    else:
                        context_words.append(sentence_words[j])

                for context_word in context_words:
                    other_words = []
                    for _ in range(k):
                        while True:
                            other_word = self.sample_word_random()
                            if other_word in context_words or other_word==word:
                                continue
                            else:
                                other_words.append(other_word)
                                break

                    batch_x.append(vocab.get_word_index(word))
                    batch_context.append(vocab.get_word_index(context_word))
                    batch_other.append([vocab.get_word_index(other_word) for other_word in other_words])

                    if(len(batch_x)>=batch_size):
                        yield (torch.tensor(batch_x, device=device),torch.tensor(batch_context, device=device),torch.tensor(batch_other, device=device))
                        batch_x.clear()
                        batch_context.clear()
                        batch_other.clear()







device = "cuda:2" if torch.cuda.is_available() else "cpu"

# vocab = Vocab.load_vocab("vocab_test")
vocab=Vocab.load_vocab("vocab_ori")
print(len(vocab))
skip_gram = SkipGram(len(vocab)).to(device)
adam = optim.Adam(skip_gram.parameters())
data_generator=DataGenerator(vocab)


epoch = 5
m = 2
k = 5
batch_size = 1024





for epoch_cnt in range(epoch):

    print("epoch", epoch_cnt)

    words_number=0
    train_time=time.time()
    for batch_iter_cnt,(batch_x, batch_context, batch_other) in enumerate(data_generator.batchIter()):
        words_number+=batch_x.size(0)

        skip_gram.train()
        adam.zero_grad()
        loss = skip_gram(batch_x, batch_context, batch_other)



        loss.backward()
        adam.step()
        if batch_iter_cnt%100==0:
            print('loss:',loss.item(),'speed :',words_number / (time.time() - train_time))

            try:
                skip_gram.eval()
                skip_gram.test_word_similar("计算机", "电脑")
                skip_gram.test_word_similar("大学", "学校")
                skip_gram.test_word_similar("数学", "瘦弱")
            except:
                pass



    #torch.save(skip_gram.state_dict(), "model" + str(epoch_cnt))