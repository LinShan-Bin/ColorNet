# “人工”智能
CLASS_NUM = 20
black = ['黑']
white = ['白', '米']
gray = ['灰']
blue = ['蓝', '兰']
red = ['红']
pink = ['粉']
green = ['绿', '牛油果', '豆沙']
dblue = ['藏青']
cyan = ['青']
purple = ['紫', '香芋']
yellow = ['黄', '焦糖']
dyellow = ['杏', '香槟', '肤色']
orenge = ['橙', '橘', '桔']
khaki = ['黄褐', '卡其']
coffee = ['咖啡', '咖', '驼']
brown = ['棕']
bw = ['黑白']
bg = ['蓝灰']
cf = ['花']
others = []

colors = [black, white, gray, blue, red, pink, green, dblue, cyan, purple, yellow, dyellow, orenge, khaki, coffee, brown, bw, bg, cf, others]

def check_color(word, color, label):
    for possible_descriptor in color:
        if possible_descriptor in word:
            return label
    return None
    
def word2label(word):
    word.replace('网红', '')
    label = None
    i = 0
    while label is None and i < CLASS_NUM:
        label = check_color(word, colors[i], i)
        i += 1
    if label is None:
        label = CLASS_NUM
        with open('others.txt', 'a') as f:
            f.write(word + '\n')
    return label


if __name__ == '__main__':
    print(word2label('网红白色长裤'))
