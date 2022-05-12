import json


class CStdLib(object):
    """
    Word2Label class
    """
    def __init__(self, cstdlib_path='./label_processor/cstdlib.json', single=True):
        with open(cstdlib_path, 'r', encoding='utf8') as f:
            cstdlib = json.load(f)
        self.remove_list = cstdlib['list']
        self.replace = cstdlib['dict']
        self.char = cstdlib['char']
        self.string = cstdlib['string']
        self.confusion = cstdlib['confusion']
        self.class_num = len(self.char) + len(self.string) + len(self.confusion)
        self.word2label = dict(zip(self.char + self.string + self.confusion, range(self.class_num)))
        self.single = single
        self.success = 0
        self.unmatched = []
        self.multi = []

    def __call__(self, org_words):
        """
        :param word: a string
        :return: int, a label
        """
        # Step 1: cut, remove and replace
        words = org_words
        for key, value in self.replace.items():
            words = words.replace(key, value)
        
        # Step 2: compare with string
        matched_str = []
        for str in self.string:
            if str in words:
                matched_str.append(str)
        
        # Step 3: compare with char
        matched_char = []
        for char in self.char:
            if char in words:
                matched_char.append(char)
        
        # Step 4: process conflicts
        remove = []
        for char in matched_char:
            for str in matched_str:
                if char in str:
                    remove.append(char)
        for char in remove:
            try:
                matched_char.remove(char)
            except:
                pass
        num_char = len(matched_char)
        num_str = len(matched_str)
        if (num_char + num_str) > 1:
            # print('Multi color detected: ', org_words, matched_char, matched_str)
            self.multi.append(org_words)
            if self.single:
                return None
            else:
                result = [self.word2label[char] for char in matched_char]
                result += [self.word2label[str] for str in matched_str]
                return result
        if self.single:
            if num_char == 1:
                self.success += 1
                return self.word2label[matched_char[0]]
            if num_str == 1:
                self.success += 1
                return self.word2label[matched_str[0]]
        else:
            if num_char == 1:
                self.success += 1
                return [self.word2label[matched_char[0]]]
            if num_str == 1:
                self.success += 1
                return [self.word2label[matched_str[0]]]
        
        # Step 4: compare with confusion
        matched_conf = []
        for conf in self.confusion:
            if conf in words:
                matched_conf.append(conf)
        if self.single:
            if len(matched_conf) == 1:
                self.success += 1
                return self.word2label[matched_conf[0]]
            # print('Multi confusion detected or No matched char/str/con: ', org_words, matched_conf)
            self.unmatched.append(org_words)
            return None
        elif len(matched_conf) > 1:
            result = [self.word2label[conf] for conf in matched_conf]
            return result
        else:
            return []


if __name__ == '__main__':
    cstdlib = CStdLib()
    label = cstdlib('黑白色')
    print(label)
    