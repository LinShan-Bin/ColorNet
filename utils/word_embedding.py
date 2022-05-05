import re
import jieba
import numpy as np
import torch
from gensim.models import KeyedVectors




class Embed():
    def __init__(self, threshold=0.45, w2v_dic='./merge_sgns_bigram_char300.txt'):
        self.threshold = threshold
        self.word2vec = KeyedVectors.load_word2vec_format(w2v_dic, binary=False)
        self.color_vec = self.word2vec['颜色']

    @staticmethod
    def white_list(word):
        return '色' in word or '绿' in word or '青' in word or '黑' in word or '白' in word or '黄' in word

    @staticmethod
    def find_chinese(file):
        # Reference: https://blog.csdn.net/bailixuance/article/details/89555580
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, '', file)
        return chinese

    def __call__(self, tags):
        embedded = []
        tags = self.find_chinese(tags)
        words = jieba.cut(tags, cut_all=False)
        for word in words:
            try:
                vec = self.word2vec[word]
                cov = np.dot(vec, self.color_vec) / (np.linalg.norm(vec) * np.linalg.norm(self.color_vec))
                if cov > self.threshold or self.white_list(word):
                    embedded.append(vec)
            except:
                continue
        embedded = torch.tensor(np.array(embedded))
        if len(embedded) < 4:
            embedded = torch.cat((embedded, torch.zeros(4 - len(embedded), 300)))
        return embedded[:4]


# if __name__ == '__main__':
#     tags = '黑色，颜色：白色'
#     embed = Embed()
#     print(embed(tags))
