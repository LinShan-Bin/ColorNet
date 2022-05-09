import re  # 正则表达式库
import collections  # 词频统计库
import jieba  # 结巴分词
import json


fp = open('./dataset/medium/train_all.json', 'r', encoding='utf8')
json_data = json.load(fp)
# for d,x in dict.items():
#     print(x['optional_tags'])
print(type(json_data))
print(len(json_data))

f = open('./word_train.txt', 'w+', encoding='utf8')

sum = 0
for key, value in json_data.items():
    for i in range(len(value['optional_tags'])):
        f.write(value['optional_tags'][i].replace("色", '')+"\n")
        sum += 1
print(sum)

fp = open('./dataset/medium/test_all.json', 'r', encoding='utf8')
json_data = json.load(fp)
# for d,x in dict.items():
#     print(x['optional_tags'])
print(type(json_data))
print(len(json_data))

f = open('./word_test.txt', 'w+', encoding='utf8')
sum = 0
for key, value in json_data.items():
    for i in range(len(value['optional_tags'])):
        f.write(value['optional_tags'][i].replace("色", '')+"\n")
        sum += 1
print(sum)


# 读取文件
fn = open('./word_train.txt', 'rt', encoding='utf-8')  # 打开文件
string_data = fn.read()  # 读出整个文件
fn.close()  # 关闭文件

# 文本预处理
pattern = re.compile(u'\t||\.|-|:|;|\)|\(|\?|"')  # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', string_data)  # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_data, cut_all=False)  # 精确模式分词
object_list = []
remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、', u'中', u'在', u'了',
                u'通常', u'如果', u'我们', u'需要', u'：', u'（', u'）', u'+', u'【', u'】', u'上衣', u'套装', u'衬衫', u'色', '连衣裙', u'外套',
                u'裤子', u'T恤', u'格子', u'裙子', u'西装', u'单件', u'条纹', u'两件套', u'长袖', u'牛仔', u'半身裙', u'马甲', u'毛衣', u'裙', u'#',
                u'短裤', u'现货', u'吊带', u'白衬衫', u'碎花', u'短袖', u'送', u'款', u'格', u'背心', u'现货', u'收藏', u'短裙', u'送礼物', u'三件套',
                u'裤', u'长裤', u'发货']  # 自定义去除词库

for word in seg_list_exact:  # 循环读出每个分词
    if word not in remove_words:  # 如果不在去除词库中
        object_list.append(word)  # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list)  # 对分词做词频统计
word_counts_top100 = word_counts.most_common(100)  # 获取前10最高频的词
print(word_counts_top100)
print(type(word_counts_top100))

color = []
for i in range(len(word_counts_top100)):
    color.append(word_counts_top100[i][0] + "色")
print(color)


fp = open('./dataset/medium/train_all.json', 'r', encoding='utf8')
json_data = json.load(fp)


B = color[:]


sum = 0
num = 0
nn = 0
mm = 0
for key, value in json_data.items():
    mm += 1
    flag = 0
    for i in range(len(value['optional_tags'])):
        num += 1
        print(num)

        ss = value['optional_tags'][i].replace("色", '')
        # 文本预处理
        pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')  # 定义正则表达式匹配模式
        ss = re.sub(pattern, '', ss)  # 将符合模式的字符去除
        # 文本分词
        ss = jieba.cut(ss, cut_all=False)  # 精确模式分词
        object_list = []
        remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、', u'中', u'在',
                        u'了', u'通常', u'如果', u'我们', u'需要', u'：', u'（', u'）', u'+', u'【', u'】', u'上衣', u'套装', u'衬衫', u'色',
                        '连衣裙', u'外套', u'裤子', u'T恤', u'格子', u'裙子', u'西装', u'单件', u'条纹', u'两件套', u'长袖', u'牛仔', u'半身裙',
                        u'马甲', u'毛衣', u'裙', u'#', u'短裤', u'现货', u'吊带', u'白衬衫', u'碎花', u'短袖', u'送', u'款', u'格', u'背心', u'现货', u'收藏', u'短裙', u'送礼物',
                        u'三件套', u'裤', u'长裤', u'发货']  # 自定义去除词库
        for word in ss:  # 循环读出每个分词
            if word not in remove_words:  # 如果不在去除词库中
                object_list.append(word+'色')  # 分词追加到列表

        flag1 = 0
        for j in range(len(B)):
            for k in range(len(object_list)):
                if B[len(B)-j-1] == object_list[k]:
                    sum += 1
                    flag1 = 1
                    break
                if j == len(B)-1 and k == len(object_list)-1:
                    flag = 1
            if flag1 == 1:
                break
    if flag == 0:
        nn += 1
print(sum)
print(nn)
print(mm)
