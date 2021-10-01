
# assumes from src
from standalone_seq2seq.pubmed_qa_dataset import PubMedQADataset
from standalone_seq2seq.simple_language_dataset import StandaloneSeq2SeqLangDataset, Seq2SeqLanguageDataset, \
    Seq2AnswerDataset
import torch
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from standalone_seq2seq.simple_encoder import LSTMLMEncoder, LSTMLMDecoder
from standalone_seq2seq.args import get_args


def predict_seq2seq(dataloader, encoder_model, decoder_model, args):
    encoder_model.eval()
    decoder_model.eval()
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        batch_ce_loss = 0.0

        num_correct = 0
        total_examples = 0
        for i, batch_package in enumerate(tqdm(dataloader)):

            x_padded, y_padded, x_lens, y_lens = batch_package

            total_examples += len(x_padded)


            x_padded = x_padded.to(args.device)
            y_padded = y_padded.to(args.device)


            final_hidden, _ = encoder_model(
                batch_package)  # we need to initialize the hidden, and get the full pass (for the built in, preconstructed rnn

            if args.repeat_hid_state:
                final_hidden = final_hidden[-1].repeat(len(final_hidden), 1, 1)

            output, _ = decoder_model.predict(batch_package,
                                              final_hidden)  # we cant argmax yet. Instead, we need to simply pass in the hidden


            encoder_model.zero_grad()
            decoder_model.zero_grad()


            flattened_y_padded = y_padded.view(-1)

            loss = criterion(output.view(len(flattened_y_padded), -1), flattened_y_padded)
            batch_ce_loss += loss.item()


            true_sequence = y_padded

            predicted_sequence = output.argmax(dim=-1).view(true_sequence.shape)  # might need to reshape further

            assert true_sequence.shape == predicted_sequence.shape

            res_tensor = true_sequence == predicted_sequence
            correct_seqs = res_tensor.all(dim=1)

            correct_hits = res_tensor.sum()
            total_hits = res_tensor.numel()  # we need to ensure we do not spuriously count the padded seq lens
            num_correct += correct_seqs.sum().item()



            summary_string = ""

        print("Epoch {} VALID loss {}\n".format(epoch, batch_ce_loss / len(dataloader)))
        print("PERFORMANCE: num correct {}; total: {}; acc: {}\n".format(num_correct,
                                                                         total_examples,
                                                                         num_correct / total_examples
                                                                         ))
        with open("{}_valid_losses.txt".format(args.run_name), "a") as file, \
                open("{}_valid_perf.txt".format(args.run_name), "a") as perf_file:

            file.write("{}\n".format(batch_ce_loss / len(dataloader)))
            perf_file.write("PERFORMANCE: num correct {}; total: {}; acc: {}\n".format(num_correct,
                                                                                       total_examples,
                                                                                       num_correct / total_examples
                                                                                       ))

    return encoder_model, decoder_model

def train_seq2answer(dataloader, encoder_model, decoder_model, args, val_dataloader):
    encoder_model.train()
    decoder_model.train()

    print("model on")
    print(next(encoder_model.parameters()).is_cuda)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    enc_optim = torch.optim.Adam(encoder_model.parameters(), lr=0.005)
    dec_optim = torch.optim.Adam(decoder_model.parameters(), lr=0.005)
    losses = []

    for epoch in range(args.epochs):
        encoder_model.train()
        decoder_model.train()
        batch_ce_loss = 0.0
        args.epoch = epoch
        # this is a batch of sequence data for our RNN to process
        for i, batch_package in enumerate(tqdm(dataloader)):
            args.batch = i

            x_padded, y_padded, x_lens, y_lens =batch_package


            x_padded = x_padded.to(args.device)
            y_padded = y_padded.to(args.device)

            final_hidden,_ = encoder_model(batch_package) #we need to initialize the hidden, and get the full pass (for the built in, preconstructed rnn

            if args.repeat_hid_state:
                final_hidden = final_hidden[-1].repeat(len(final_hidden),1,1)




            # sample some random noise and then choose whether to do teacher_forcing or not
            teacher_forcing = torch.empty(1).uniform_(0,1)


            output, _ = decoder_model(batch_package,
                                      final_hidden,
                                      teacher_force=teacher_forcing < args.teacher_force_ratio)  # we cant argmax yet. Instead, we need to simply pass in the hidden


            encoder_model.zero_grad()
            decoder_model.zero_grad()
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            y_padded = y_padded.view(-1)
            loss = criterion(output.view(len(y_padded), -1), y_padded.view(-1))  # our loss is not exactly this...
            batch_ce_loss += loss.item()
            loss.backward()
            enc_optim.step()
            dec_optim.step()



        losses.append(batch_ce_loss/len(dataloader))

        # do some validation testing
        with open("{}_losses.txt".format(args.run_name), "a") as file:
            file.write("{}\n".format(batch_ce_loss/len(dataloader)))

        predict_seq2seq(val_dataloader, encoder_model, decoder_model, args)
        print("Epoch {} loss {}\n".format(epoch, batch_ce_loss/len(dataloader)))


    torch.save({

        'encoder': encoder_model.state_dict(),
        'decoder': decoder_model.state_dict(),
        'enc_optim': enc_optim.state_dict(),
        'dec_optim': dec_optim.state_dict()
    }, "{}_experiment.pt".format(args.run_name))

    return encoder_model, decoder_model, enc_optim, dec_optim

'''
Training method
'''
def train_seq2seq(dataloader, encoder_model, decoder_model, args, val_dataloader):
    encoder_model.train()
    decoder_model.train()

    print("model on")
    print(next(encoder_model.parameters()).is_cuda)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    enc_optim = torch.optim.Adam(encoder_model.parameters(), lr=0.005)
    dec_optim = torch.optim.Adam(decoder_model.parameters(), lr=0.005)
    losses = []

    for epoch in range(args.epochs):
        encoder_model.train()
        decoder_model.train()
        batch_ce_loss = 0.0
        args.epoch = epoch
        # this is a batch of sequence data for our RNN to process
        for i, batch_package in enumerate(tqdm(dataloader)):
            args.batch = i

            x_padded, y_padded, x_lens, y_lens =batch_package


            x_padded = x_padded.to(args.device)
            y_padded = y_padded.to(args.device)

            final_hidden,_ = encoder_model(batch_package) #we need to initialize the hidden, and get the full pass (for the built in, preconstructed rnn

            if args.repeat_hid_state:
                final_hidden = final_hidden[-1].repeat(len(final_hidden),1,1)




            # sample some random noise and then choose whether to do teacher_forcing or not
            teacher_forcing = torch.empty(1).uniform_(0,1)


            output, _ = decoder_model(batch_package,
                                      final_hidden,
                                      teacher_force=teacher_forcing < args.teacher_force_ratio)  # we cant argmax yet. Instead, we need to simply pass in the hidden


            encoder_model.zero_grad()
            decoder_model.zero_grad()
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            y_padded = y_padded.view(-1)
            loss = criterion(output.view(len(y_padded), -1), y_padded.view(-1))  # our loss is not exactly this...
            batch_ce_loss += loss.item()
            loss.backward()
            enc_optim.step()
            dec_optim.step()



        losses.append(batch_ce_loss/len(dataloader))

        # do some validation testing
        with open("{}_losses.txt".format(args.run_name), "a") as file:
            file.write("{}\n".format(batch_ce_loss/len(dataloader)))

        predict_seq2seq(val_dataloader, encoder_model, decoder_model, args)
        print("Epoch {} loss {}\n".format(epoch, batch_ce_loss/len(dataloader)))


    torch.save({

        'encoder': encoder_model.state_dict(),
        'decoder': decoder_model.state_dict(),
        'enc_optim': enc_optim.state_dict(),
        'dec_optim': dec_optim.state_dict()
    }, "{}_experiment.pt".format(args.run_name))

    return encoder_model, decoder_model, enc_optim, dec_optim

def pad_collate(batch):
    from torch.nn.utils.rnn import pad_sequence

    (xx,yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value =0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value =0)

    return xx_pad, yy_pad, x_lens, y_lens

'''
Runs seq2seq
'''
def seq2seq():
    import os
    args = get_args()
    # Seq2SeqLanguageDataset
    # train_dataset = StandaloneSeq2SeqLangDataset(path="data/ori_pqal.json")

    path_offset = "../data/vqa/"
    sources = ["train.json", "minival.json", "nominival.json"]
    paths = list(map(lambda x: os.path.join(path_offset, x), sources))

    total_dataset = Seq2SeqLanguageDataset(paths=paths)
    args.language = total_dataset.language

    print(paths)
    train_dataset = Seq2SeqLanguageDataset(paths=[paths[0]])
    train_dataset.set_language(args.language)
    val_dataset = Seq2SeqLanguageDataset(paths=[paths[1]])
    val_dataset.set_language(args.language)

    for key, val in train_dataset.stats.items():
        print("{} {}".format(key, val))

    for key, val in val_dataset.stats.items():
        print("{} {}".format(key, val))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bs,
        shuffle=args.shuffle, num_workers=4,  # set numworkers lower to get it to work
        drop_last=args.drop_last, pin_memory=True, collate_fn=pad_collate
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.bs,
        shuffle=args.shuffle, num_workers=4,  # set numworkers lower to get it to work
        drop_last=args.drop_last, pin_memory=True, collate_fn=pad_collate
    )

    encoder_model = LSTMLMEncoder(embedding_dim=args.embedding_dim, hidden_size=args.rnn_hidden_size,
                                  vocab_size=args.language.get_language_size(), args=args,
                                  num_rnn_layers=args.rnn_layers,
                                  bidirectional=args.bidirectional).to(device=args.device)


    decoder_model = LSTMLMDecoder(embedding_dim=args.embedding_dim, hidden_size=args.rnn_hidden_size,
                                  vocab_size=args.language.get_language_size(), args=args, language=args.language,
                                  num_rnn_layers=args.rnn_layers,
                                  bidirectional_encoder=args.bidirectional).to(device=args.device)
    train_seq2seq(train_dataloader, encoder_model, decoder_model, args, val_dataloader)
    print(train_dataset[100])
    print(train_dataset.view_data(100)["QUESTION"])
    print(train_dataset.view_data(100)["LONG_ANSWER"])

    # print(my_obj[100]["QUESTION"])
    # print(my_obj[100]["final_decision"])
    # print(my_obj[100]["LONG_ANSWER"])
    pass

def seq2ans():
    import os
    args = get_args()
    # Seq2SeqLanguageDataset


    path_offset = "data/"
    sources = ["ori_pqaa.json", "ori_pqal.json"]
    paths = list(map(lambda x: os.path.join(path_offset, x), sources))

    total_dataset = PubMedQADataset(paths=paths)
    args.language = total_dataset.language
    # print(total_dataset.answer_list)

    # print(paths)
    train_dataset = PubMedQADataset(paths=[paths[0]])
    train_dataset.set_language(args.language)
    val_dataset = PubMedQADataset(paths=[paths[1]])
    val_dataset.set_language(args.language)

    for key, val in train_dataset.stats.items():
        print("{} {}".format(key, val))

    for key, val in val_dataset.stats.items():
        print("{} {}".format(key, val))


    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bs,
        shuffle=args.shuffle, num_workers=4,  # set numworkers lower to get it to work
        drop_last=args.drop_last, pin_memory=True, collate_fn=pad_collate
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.bs,
        shuffle=args.shuffle, num_workers=4,  # set numworkers lower to get it to work
        drop_last=args.drop_last, pin_memory=True, collate_fn=pad_collate
    )
    # consider how the embedding dim is connected back to the tensors themselves.
    # how can we ensure words are actually mapped back to it ?
    encoder_model = LSTMLMEncoder(embedding_dim=args.embedding_dim, hidden_size=args.rnn_hidden_size,
                                  vocab_size=args.language.get_language_size(), args=args,
                                  num_rnn_layers=args.rnn_layers,
                                  bidirectional=args.bidirectional).to(device=args.device)
    decoder_model = LSTMLMDecoder(embedding_dim=args.embedding_dim, hidden_size=args.rnn_hidden_size,
                                  vocab_size=args.language.get_language_size(), args=args, language=args.language,
                                  num_rnn_layers=args.rnn_layers,
                                  bidirectional_encoder=args.bidirectional).to(device=args.device)
    train_seq2seq(train_dataloader, encoder_model, decoder_model, args, val_dataloader)



if __name__  == "__main__":
    seq2ans()

