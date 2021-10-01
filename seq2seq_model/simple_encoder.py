
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim

# try using gensim, FastText vs Glove vectors
# first, try using fasttext
class LSTMLMEncoder(nn.Module):

    def __init__(self,  embedding_dim, hidden_size, vocab_size, args, num_rnn_layers=2, bidirectional=False):
        super(LSTMLMEncoder, self).__init__()
        self.hidden_dim = hidden_size
        self.bidirectional = bidirectional
        print("bidifirectional?")
        print(self.bidirectional)
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)

        if args.use_pretrained_token_embeddings and args.pretrained_token_embeddings_path:
            print("loading the fasttext vectors. Note that we can save the torch matrix itself")

            '''
            we need to be very careful here, since we need to set the 
            words appropriately. That is: ensure that our language/vocab-mapping
            also corresponds to the real stuff
            '''

            model = gensim.models.KeyedVectors.load_word2vec_format(
                args.pretrained_token_embeddings_path)

            weights_matrix = torch.empty((args.language.get_language_size(),
                                                embedding_dim))


            '''
            Good old fashioned loading.

            token_to_label
            label_to_token
            '''
            words_found = 0

            for ind, token in args.language.label_to_token.items():
                if token not in model:
                    # print("oops!")
                    continue
                else:
                    weights_matrix[ind] = torch.FloatTensor(model[token]).view(1, embedding_dim)
                    words_found += 1

            print("{}/{} words warmstarted".format(words_found, args.language.get_language_size()))
            self.embed_layer.from_pretrained(weights_matrix)

        self.num_rnn_layers=  num_rnn_layers
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=self.num_rnn_layers,
                          batch_first=True, bidirectional=self.bidirectional)  # this will be useful later

        # we might need to pad_packed_seq right here! due to taking the hidden dim.
        # in fact, we could make this all native to the RNN itself
        self.vocab_layer = nn.Linear(hidden_size, vocab_size)
        self.args =args


    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.num_rnn_layers, self.args.bs, self.hidden_dim)
        # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)


            # hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        # hidden_b = Variable(hidden_b)

        return (hidden_a.cuda())

    def forward(self, batch_package):
        self.hidden = self.init_hidden()

        if self.bidirectional:
            # self.hidden = torch.stack((self.hidden, self.hidden),dim=0)
            self.hidden = torch.cat((self.hidden, self.hidden), dim=0)

        '''need to unpack the sequence!'''
        (x_padded, y_padded, x_lens, y_lens ) = batch_package
        x_padded = x_padded.to(self.args.device)
        # print("k, here it is")
        # print(x_padded.shape)
        y_padded = y_padded.to(self.args.device)




        x = self.embed_layer(x_padded)
        '''now, make sure we pack'''
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True,
                                        enforce_sorted=False)
        x_packed.cuda()
        self.hidden.cuda()

        # print(x_packed.is_cuda)
        # print(self.hidden.is_cuda)

        rnn_out, (h_n) = self.rnn(x_packed.to(self.args.device),
                                self.hidden.to(self.args.device))
        # print("hidden shapes")
        # print(h_n.shape)
        # print(rnn_out.data.shape)
        seq_predicts,_ = pad_packed_sequence(rnn_out,batch_first=True)
        # what can we do with a padded sequence
        # print(seq_predicts)
        # seq predicts recovers the entire set of hidden states. in theory,
        # we should have the final hidden state of this equal to the final hidden state already returned
        # print(seq_predicts.data.shape) # just returns the pad packed sequence => BATCH_SUM_SEQ * num feats
        # print(seq_predicts.shape)
        return h_n,h_n # return the logits

'''
Cant have a voacb layer here'''
class LSTMLMDecoder(nn.Module):

    def __init__(self,  embedding_dim, hidden_size, vocab_size, args,
                 language, num_rnn_layers=2, bidirectional_encoder=False):
        super(LSTMLMDecoder, self).__init__()
        self.hidden_dim = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim)
        if bidirectional_encoder:
            # print("gotten called")
            self.hidden_dim *= 1

        self.num_rnn_layers = num_rnn_layers


        if args.use_pretrained_token_embeddings and args.pretrained_token_embeddings_path:
            print("loading the fasttext vectors. Note that we can save the torch matrix itself")

            '''
            we need to be very careful here, since we need to set the 
            words appropriately. That is: ensure that our language/vocab-mapping
            also corresponds to the real stuff
            '''

            model = gensim.models.KeyedVectors.load_word2vec_format(
                args.pretrained_token_embeddings_path)

            weights_matrix = torch.empty((args.language.get_language_size(),
                                                embedding_dim))


            '''
            Good old fashioned loading.

            token_to_label
            label_to_token
            '''
            words_found = 0

            for ind, token in args.language.label_to_token.items():
                if token not in model:
                    # print("oops!")
                    continue
                else:
                    weights_matrix[ind] = torch.FloatTensor(model[token]).view(1, embedding_dim)
                    words_found += 1

            print("{}/{} words warmstarted".format(words_found, args.language.get_language_size()))
            self.embed_layer.from_pretrained(weights_matrix)

        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_dim,
                      num_layers=self.num_rnn_layers,
                      batch_first=True)  # this will be useful later
        # we might need to pad_packed_seq right here! due to taking the hidden dim.
        # in fact, we could make this all native to the RNN itself
        self.vocab_layer = nn.Linear(self.hidden_dim, vocab_size)
        self.args =args
        self.language = language




    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.num_rnn_layers, self.args.bs, self.hidden_dim)
        # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)


            # hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        # hidden_b = Variable(hidden_b)

        return (hidden_a.cuda())

    def predict(self, batch_package, final_hidden):

        (x_padded, y_padded, x_lens, y_lens) = batch_package
        x_padded = x_padded.to(self.args.device)
        y_padded = y_padded.to(self.args.device)

        x_0 = self.language.convert_tokens_to_labels_lm("", sent_start=True)
        top_letter = x_0.to(self.args.device)
        top_letter = top_letter.repeat(len(x_padded)) #repeat k times
        top_letter.unsqueeze_(1)
        # print(top_letter.shape)
        h_n = final_hidden

        if self.bidirectional_encoder:
            # print("gettin gher")
            h_n = h_n.view(self.num_rnn_layers, h_n.shape[1], self.hidden_dim)

        # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        # hidden_b = hidden_b.cuda()

        # h_n = Variable(h_n)
        # h_n = h_n.to(self.args.device)

        # total_word = ""

        word_tensors = torch.empty(size=(y_padded.shape[0],
                                         y_padded.shape[1],
                                         self.language.vocab_size)
                                         ).to(self.args.device)
        # word_tensors.unsqueeze_(-1)
        print("WT SHAPE (predict)")

        # if we're dealing with characters ~20; else if we deal with word ~5 seq len

        # BS * 5 * 102
        # BS * 20 * 17532 (vocab size)

        print(word_tensors.shape)

        '''steps: we just need to do: decode the letters each time. 

        should be simple
        and also do torch argmax correctly over dimensions etc. and might need to do a reshape
        might need to deal with batches too
        '''

        for i in range(y_padded.shape[1]):
            new_x = self.embed_layer(top_letter)
            '''each should be BS * SEQ_LEN * hidden_dim size. Then, we embed directly
             which takes BS * (1) * hidden_dim_size. => BS * 1 *  VOCAB 
             '''
            # print(new_x.shape)
            top_letter, h_n = self.rnn(new_x.to(self.args.device),
                                       h_n.to(self.args.device))
            next_letter_predictions = self.vocab_layer(top_letter)
            top_letter = next_letter_predictions.argmax(dim=-1)
            # print(top_letter.shape)
            # predict_list_of_letters = self.language.convert_labels_to_letters(top_letter)
            word_tensors[:,i, :] = next_letter_predictions.squeeze()

        word_predictions = word_tensors.argmax(-1)
        list_of_questions = self.language.convert_labels_to_tokens(x_padded)
        list_of_answers = self.language.convert_labels_to_tokens(y_padded)
        list_of_outputs = self.language.convert_labels_to_tokens(word_predictions)
        with open("{}_results.txt".format(self.args.run_name), "a") as file:
            for tup in zip(list_of_questions, list_of_answers, list_of_outputs):
                file.write("{} , {} , {}\n".format(tup[0],tup[1],tup[2]))

        # print(word_tensors.shape)



        # for i in range(23):
        #     input_question = self.language.convert_labels_to_letters(x_padded[i].unsqueeze_(1))
        #     input_answer = self.language.convert_labels_to_letters((y_padded[i]).unsqueeze_(1))
        #     print(input_question, input_answer)
        # # print(x_padded)
        #
        # print(total_word)
        # return total_word, ""
        return word_tensors, word_predictions

    # TODO: change this train_mode to teacher force. Or understand what is going on
    def forward(self, batch_package, hidden, teacher_force=False):



        self.hidden = hidden # the hidden will be the Encoder outputs..
        # print(self.hidden.shape)
        if self.bidirectional_encoder:
            # print("gettin gher")
            self.hidden = self.hidden.view(self.num_rnn_layers, self.hidden.shape[1], self.hidden_dim)
            # print(self.hidden.shape)

        (x_padded, y_padded, x_lens, y_lens) = batch_package

        if not teacher_force: #non teacher forced version
            # print(self.hidden.info)
            # print(self.hidden.shape)
            '''need to unpack the sequence!'''
            x_padded = x_padded.to(self.args.device)
            y_padded = y_padded.to(self.args.device)



            y = self.embed_layer(y_padded) # THE EMBED LAYER IS USED!
            '''now, make sure we pack'''
            y_packed = pack_padded_sequence(y, y_lens, batch_first=True,
                                            enforce_sorted=False)

            # print("hidden shape (decoder)")
            # print(hidden.shape)
            rnn_out, h_n = self.rnn(y_packed.to(self.args.device),
                                    self.hidden.to(self.args.device))

            seq_predicts,_ = pad_packed_sequence(rnn_out,batch_first=True)
            return self.vocab_layer(seq_predicts),h_n # return the logits
        else:
                #'''teacher forced version'''
            x_0 = self.language.convert_tokens_to_labels_lm("", sent_start=True)
            x_0 = x_0.repeat(len(x_padded))
            top_letter = x_0.to(self.args.device)
            top_letter.unsqueeze_(1)
            # print(top_letter.shape)
            h_n = self.hidden
             # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)


            # hidden_b = hidden_b.cuda()

            # h_n = Variable(h_n)
            # h_n = h_n.to(self.args.device)

            word_tensors = torch.empty(size=y_padded.shape).to(self.args.device)
            word_tensors.unsqueeze_(-1)
            word_tensors = word_tensors.repeat(1,1,self.language.vocab_size)
            # print("WT SHAPE")
            # print(word_tensors.shape)

            '''steps: we just need to do: decode the letters each time. 
            
            should be simple
            and also do torch argmax correctly over dimensions etc. and might need to do a reshape
            might need to deal with batches too
            '''

            for i in range(y_padded.shape[1]):
                new_x = self.embed_layer(top_letter)
                '''each should be BS * SEQ_LEN * hidden_dim size. Then, we embed directly
                 which takes BS * (1) * hidden_dim_size. => BS * 1 *  VOCAB 
                 '''
                # print(new_x.shape)
                next_letter_predictions, h_n = self.rnn(new_x.to(self.args.device),
                                        h_n.to(self.args.device))

                next_letter_predictions = self.vocab_layer(next_letter_predictions)
                top_letter = next_letter_predictions.argmax((-1))


                # print("top_letter shape")
                # print(top_letter.shape)

                word_tensors[:,i, :] = next_letter_predictions.squeeze() #column-wise assignment

                # predict_list_of_letters = self.language.convert_labels_to_letters(top_letter)
                # total_word +=predict_list_of_letters[0]

            # log_res = torch.empty((1)).uniform_(0,1)
            # if log_res > 0.99:
            if self.args.batch == 0:

                # print(total_word)
                # print("SHAPE")
                # print(word_tensors.shape)
                list_of_questions = self.language.convert_labels_to_tokens(x_padded)
                list_of_answers = self.language.convert_labels_to_tokens(y_padded)
                list_of_outputs = self.language.convert_labels_to_tokens(word_tensors.argmax(dim=-1))
                with open("{}_teacher_forced_results.txt".format(self.args.run_name), "a") as file:
                    file.write("="*75 + "\n")
                    file.write("Epoch {}, Batch {}\n".format(self.args.epoch, self.args.batch))
                    for tup in zip(list_of_questions, list_of_answers, list_of_outputs):
                        file.write("{} , {} , {}\n".format(tup[0], tup[1], tup[2]))
            # word_tensors.squeeze_(-1)

            return word_tensors,h_n
