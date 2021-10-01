import collections

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import re

from standalone_seq2seq.language_model.simple_language import Language

'''A stand-alone seq2seq dataset. 
It should accept a path for where the data is located, then load it in
The goal is to simply make it take in the JSON only.
Then, in that case, we should make it diff.
PubmedQA seems like we should take both U and A; we can deal with any of them!

This code will both construct the vocab as well as return the pairs.

This code will construct the language, but should also be able to accept one


'''
import json
from collections import defaultdict
'''Consider making this dataset using TorchText in the future'''
class StandaloneSeq2SeqLangDataset(Dataset):
    '''
    construct with the path to the json data
    '''
    def __init__(self, path):
        '''incredibly, in python we can load JSON with a single call'''
        '''now, data has the initial json stuff'''
        self.data = []
        self.json_obj = json.load(open(path))
        self.qa_data = []
        self.stats = defaultdict(int)

        self.vocab_list = set()
        for key, datum in self.json_obj.items():
            self.data.append(datum)

            ques = datum["QUESTION"]

            # re.match and re.search() and then all else we need is simply: how to replace. So stuff like:
            # re.findall and re.replace etc.
            ques = re.sub("[^a-zA-Z0-9\s]", "", ques)
            ans = datum["final_prediction"]
            ans = re.sub("[^a-zA-Z0-9\s]", "", ans)
            # print(ques, ans)

            tokenized_ques = ques.split()
            tokenized_ans = ans.split()


            self.vocab_list.update(tokenized_ques)
            self.vocab_list.update(tokenized_ans)
            content_dict = {"QUESTION": tokenized_ques, "LONG_ANSWER":tokenized_ans}
            self.qa_data.append(content_dict)

            self.stats["q_tokens"] += len(tokenized_ques)
            self.stats["a_tokens"] += len(tokenized_ans)
            self.stats["max_q_len"] = max(self.stats["max_q_len"] , len(tokenized_ques))
            self.stats["max_a_len"] = max(self.stats["max_a_len"] , len(tokenized_ans))



        print(len(self.data))

        self.stats["total"] = len(self.data)


        self.language = Language(word_level=True, vocab_list=self.vocab_list)

    def set_language(self, language: Language):
        self.language = language

    def view_data(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len (self.data)

    def __getitem__(self, idx: int):
        return self.language.convert_tokens_to_labels_lm(self.qa_data[idx]["QUESTION"]),\
            self.language.convert_tokens_to_labels_lm(self.qa_data[idx]["LONG_ANSWER"])





class Seq2SeqLanguageDataset(Dataset):
    def __init__(self,  paths: list):
        from collections import defaultdict

        super().__init__()
        import os

        self.data = []

        self.qa_data = []
        self.vocab_list = set()
        self.stats = defaultdict(int)


        for path in paths:
            print("processing")
            print(os.path.abspath(path))

            # json obj is list with thousands of items
            self.json_obj = json.load(open(path))

            for datum in self.json_obj:
                if len(datum["label"]) > 0:
                    self.data.append(datum) # we only need the sentence data now

                    ques = datum["sent"]

                    # re.match and re.search() and then all else we need is simply: how to replace. So stuff like:
                    # re.findall and re.replace etc.
                    ques = re.sub("[^a-zA-Z0-9\s]", "", ques)
                    ans = list(datum['label'].keys())[0]
                    ans = re.sub("[^a-zA-Z0-9\s]", "", ans)

                    tokenized_ques = ques.split()
                    tokenized_ans = ans.split()

                    self.vocab_list.update(tokenized_ques)
                    self.vocab_list.update(tokenized_ans)
                    content_dict = {"sent": tokenized_ques, "label": tokenized_ans}
                    self.qa_data.append(content_dict)

                    self.stats["q_tokens"] += len(tokenized_ques)
                    self.stats["a_tokens"] += len(tokenized_ans)
                    self.stats["max_q_len"] = max(self.stats["max_q_len"], len(tokenized_ques))
                    self.stats["max_a_len"] = max(self.stats["max_a_len"], len(tokenized_ans))

            print("Use %d data in torch dataset" % (len(self.data)))

        self.language = Language(word_level=True, vocab_list=self.vocab_list)


    def set_language(self, language: Language):
        self.language = language

    def get_vocab_size(self):
        return self.language.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        '''add SOS to X 
        and EOS to Y
        '''
        ques = datum['sent'].split()
        ans = list(datum['label'].keys())[0].split()
        ques = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ques)
        ans = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ans)



        return self.language.convert_tokens_to_labels_lm(ques, sent_start=True, sent_end=True),\
               self.language.convert_tokens_to_labels_lm(ans, sent_start=True, sent_end=True)


class Seq2AnswerDataset(Dataset):
    def __init__(self,  paths: list):
        from collections import defaultdict

        super().__init__()
        import os

        self.data = []

        self.qa_data = []
        self.question_vocab_list = set()
        self.answer_list = set()

        self.stats = defaultdict(int)


        for path in paths:
            print("processing")
            print(os.path.abspath(path))

            # json obj is list with thousands of items
            self.json_obj = json.load(open(path))

            for key,datum in self.json_obj.items():
                # if len(datum["label"]) > 0:
                self.data.append(datum) # we only need the sentence data now

                ques = datum["QUESTION"]

                # re.match and re.search() and then all else we need is simply: how to replace. So stuff like:
                # re.findall and re.replace etc.
                ques = re.sub("[^a-zA-Z0-9\s]", "", ques)
                ans = datum['final_decision']


                ans = re.sub("[^a-zA-Z0-9\s]", "", ans)

                tokenized_ques = ques.split()
                tokenized_ans = ans.split()

                self.question_vocab_list.update(tokenized_ques)
                self.answer_list.update(tokenized_ans)
                content_dict = {"QUESTION": tokenized_ques, "final_decision": tokenized_ans}
                self.qa_data.append(content_dict)

                self.stats["q_tokens"] += len(tokenized_ques)
                self.stats["a_tokens"] += len(tokenized_ans)
                self.stats["max_q_len"] = max(self.stats["max_q_len"], len(tokenized_ques))
                self.stats["max_a_len"] = max(self.stats["max_a_len"], len(tokenized_ans))

            print("Use %d data in torch dataset" % (len(self.data)))


        self.language = Language(word_level=True, vocab_list=self.question_vocab_list)


    def set_language(self, language: Language):
        self.language = language

    def get_vocab_size(self):
        return self.language.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        '''add SOS to X 
        and EOS to Y
        '''
        ques = datum['QUESTION'].split()

        ans = datum["final_decision"].split()
        ques = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ques)
        ans = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ans)



        return self.language.convert_tokens_to_labels_lm(ques, sent_start=True),\
               self.language.convert_tokens_to_labels_lm(ans, sent_start=True, sent_end=True)
