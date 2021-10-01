import collections

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import re

from standalone_seq2seq.language_model.simple_language import Language
import json
from collections import defaultdict
class PubMedQADataset(Dataset):
    def __init__(self,  paths: list):
        from collections import defaultdict

        super().__init__()
        import os

        self.data = []

        self.qa_data = []
        self.vocab_list = set()
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

                self.vocab_list.update(tokenized_ques)
                self.vocab_list.update(tokenized_ans)
                self.answer_list.update(tokenized_ans)
                content_dict = {"QUESTION": tokenized_ques, "final_decision": tokenized_ans}
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
        ques = datum['QUESTION'].split()

        ans = datum["final_decision"].split()
        ques = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ques)
        ans = map(lambda elt: re.sub("[^a-zA-Z0-9\s]", "", elt) , ans)



        return self.language.convert_tokens_to_labels_lm(ques, sent_start=True),\
               self.language.convert_tokens_to_labels_lm(ans, sent_start=True, sent_end=True)
