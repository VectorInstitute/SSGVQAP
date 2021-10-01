from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import numpy as np
'''

The Language class provides a conversion from string to a tensor representation.

For now: given ("text") => [201 100 93 211]

i.e. turn a text into a sequence of characters/labels
'''
class Language:



    def __init__(self, word_level=False, vocab_list = None):
        self.word_level = word_level
        if self.word_level == True and vocab_list is None:
            raise Exception("Must specify vocab list if word level language")
        if word_level:
            self.all_tokens = list(vocab_list)
            self.all_tokens.insert( 0, "") #add the empty; it should be at 0 index, in order to be correctly ignored.
        else:
            self.all_tokens = string.printable
                         #+ 1 + 1 # Plus SOS and EOS marker

        self.token_to_label = {
            letter: ind for (ind, letter) in enumerate(self.all_tokens)

        } #maps from "s" => 23
        self.token_to_label["<BOS>"] = len(self.token_to_label) #this actually works! (since we add an additional key!
        self.token_to_label["<EOS>"] = len(self.token_to_label)

        self.label_to_token = {
            ind:  letter for ( letter, ind) in self.token_to_label.items()

        } #maps from 23 => "s"

        self.vocab_size = len(self.token_to_label)
        # self.vocab_size = len(self.word_to_label)

        pass
    def get_language_size(self):
        assert len(self.token_to_label) == len(self.label_to_token)
        return len(self.token_to_label)

    def convert_tokens_to_labels(self, inp_string):
        target_tensor = torch.empty((len(inp_string)), dtype=torch.long)
        for curr_ind,char in enumerate(inp_string):
            target_tensor[curr_ind] = self.token_to_label[char]
        return target_tensor

    '''
        now, this takes an iteratble
        '''
    def convert_tokens_to_labels_lm(self, inp_token_list, sent_start=False, sent_end=False):
        temp_lst = []

        # target_tensor = torch.empty((len(inp_string)+1), dtype=torch.long)
        for curr_ind,char in enumerate(inp_token_list):
            temp_lst.append(self.token_to_label[char])

        if sent_start:
            temp_lst.insert(0, self.token_to_label["<BOS>"])
        if sent_end:
            temp_lst.append(self.token_to_label["<EOS>"])

        # print(temp_lst)

        return torch.tensor(temp_lst, dtype=torch.long)

    # def my_func(cls, np_arry):
    #
    #     for elt in np_arry:
    #
    #     pass

    # def convert_labels_to_letters_np(self, np_arry):
    #     np.apply_along_axis()
    #     pass


    '''assuming BS X seq len X 1
    returns a list of words decoded.
    return style is BS X [Str] , where length of str is determined appropriately (should be seq len)!
    can we use numpy instead?
    '''
    def convert_labels_to_tokens(self, inp_tensor):
        # print(inp_tensor.shape)
        target_strings = [[] for i in range(len(inp_tensor))]
        # print(len(target_strings))
        delim_char = " " if self.word_level else ""
        for ind, elt in enumerate(inp_tensor):
            for curr_ind, label in enumerate(inp_tensor[ind]):
                target_strings[ind].append(self.label_to_token[label.item()])


            target_strings[ind] = delim_char.join(target_strings[ind])

        return target_strings

    '''assuming BS X seq len X 1
    returns a list of words decoded.
    return style is BS X [Str] , where length of str is determined appropriately (should be seq len)!
    can we use numpy instead?
    '''
    def convert_labels_to_letters_vectorized(self, inp_tensor):
        '''logic would essentially be the same tbh.
        inp tensor => numpy array => apply string function (via apply along axis or vectorize) => convert back to string
        '''
        numpy_str_array = np.empty(shape= inp_tensor.shape, dtype="<U{}".format(inp_tensor.shape[1]))
        print(inp_tensor.shape)
        target_strings = [[] for i in range(len(inp_tensor))]
        print(len(target_strings))
        for ind, elt in enumerate(inp_tensor):
            for curr_ind, label in enumerate(inp_tensor[ind]):
                target_strings[ind].append(self.label_to_token[label.item()])
            target_strings[ind] = ''.join(target_strings[ind])
        return target_strings


    def findFiles(path): return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_tokens
        )

    # Define a method which converts from a string, to the tokens

