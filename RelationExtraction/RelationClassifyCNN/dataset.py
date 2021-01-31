import os
import torch
import numpy as np
import json
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader


class processor(object):
    def __init__(self):
        pass

    '''text cleansing'''

    def search_entity(self, sentence):
        # extract the entities
        e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
        e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]

        sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
        sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)

        sentence = word_tokenize(sentence)
        sentence = ' '.join(sentence)
        sentence = sentence.replace('< e1 >', '<e1>')
        sentence = sentence.replace('< e2 >', '<e2>')
        sentence = sentence.replace('< /e1 >', '</e1>')
        sentence = sentence.replace('< /e2 >', '</e2>')
        sentence = sentence.split()

        assert '<e1>' in sentence
        assert '<e2>' in sentence
        assert '</e1>' in sentence
        assert '</e2>' in sentence

        ## two entiti location index finding
        subj_start = subj_end = obj_start = obj_end = 0

        pure_sentence = []
        for i, word in enumerate(sentence):
            if '<e1>' == word:
                subj_start = len(pure_sentence)
                continue
            if '</e1>' == word:
                subj_end = len(pure_sentence) - 1
                continue
            if '<e2>' == word:
                obj_start = len(pure_sentence)
                continue
            if '</e2>' == word:
                obj_end = len(pure_sentence) - 1
                continue
            pure_sentence.append(word)
        return e1, e2, subj_start, subj_end, obj_start, obj_end, pure_sentence

    '''covert to be json format'''

    def convert(self, path_src, path_des,sample_size=None):
        with open(path_src, 'r', encoding='utf-8') as fr:
            if sample_size is None:
                data = fr.readlines()
            else:
                data = fr.readlines()[:sample_size]

        with open(path_des, 'w', encoding='utf-8') as fw:
            for i in tqdm(range(0, len(data), 4)):
                id_s, sentence = data[i].strip().split('\t')
                sentence = sentence[1:-1]
                e1, e2, subj_start, subj_end, obj_start, obj_end, sentence = self.search_entity(sentence)
                meta = dict(
                    id=id_s,
                    relation=data[i + 1].strip(),
                    head=e1,
                    tail=e2,
                    subj_start=subj_start,
                    subj_end=subj_end,
                    obj_start=obj_start,
                    obj_end=obj_end,
                    sentence=sentence,
                    comment=data[i + 2].strip()[8:]
                )
                json.dump(meta, fw, ensure_ascii=False)
                fw.write('\n')


# Load the word embeeding
class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        #
        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))
        if ("*UNKNOWN*" not in word2id):
            word2id['*UNKNOWN*'] = len(word2id)
            unk_emb = np.random.uniform(-1, 1, self.word_dim)
            word_vec.append(unk_emb)
        pad_emb = np.zeros([1, self.word_dim], dtype=np.float32)  # <pad> is initialize as zero
        word_vec = np.concatenate((pad_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec



class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class SemEvalDateset(Dataset):
    def __init__(self, filename, rel2id, word2id, config):
        self.filename = filename
        self.rel2id = rel2id
        self.word2id = word2id
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    # position encoding
    # pos_dis: parameter for position encoding: the length to shift the position
    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    # relative position encoding
    def __get_relative_pos(self, x, entity_pos):
        # entity_pos[0] -> begin
        # entity_pos[1] -> end
        if x < entity_pos[0]:
            return self.__get_pos_index(x - entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x - entity_pos[1])
        else:
            return self.__get_pos_index(0)

    # sentence feature
    def _symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """

        mask = [1] * len(sentence)
        # for exmaple  [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
        # 1 for the positions before the first entity
        # 2 for the positions bwt first ane second entity
        # 3 for the positions after the second entity

        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1] + 1):
                mask[i] = 2
            for i in range(e2_pos[1] + 1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1] + 1):
                mask[i] = 2
            for i in range(e1_pos[1] + 1, len(sentence)):
                mask[i] = 3

        words = []  # words id list
        pos1 = []  # word position relative to es1
        pos2 = []  # word pisition relative to es2

        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i], self.word2id['*UNKNOWN*']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        # PADDING
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    # lexical feature
    def _lexical_feature(self, e1_idx, e2_idx, sent):

        def _entity_context(e_idx, sent):
            ''' return [w(e-1), w(e), w(e+1)]
            '''
            context = []
            context.append(sent[e_idx])
            if e_idx >= 1:
                context.append(sent[e_idx - 1])
            else:
                context.append(sent[e_idx])

            if e_idx < len(sent) - 1:
                context.append(sent[e_idx + 1])
            else:
                context.append(sent[e_idx])
            return context

        # to find the right and left word
        context1 = _entity_context(e1_idx[0], sent)
        context2 = _entity_context(e2_idx[0], sent)

        # ignore WordNet hypernyms in paper
        lexical = context1 + context2
        #         print(sent)
        #         print(lexical)
        lexical_ids = [self.word2id.get(word, self.word2id['*UNKNOWN*']) for word in lexical]
        lexical_ids = np.asarray(lexical_ids, dtype=np.int64)
        #         print(lexical_ids)
        return np.reshape(lexical_ids, newshape=(1, 6))

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self._symbolize_sentence(e1_pos, e2_pos, sentence)

                lexical = self._lexical_feature(e1_pos, e2_pos, sentence)

                temp = (one_sentence, lexical)
                data.append(temp)
                # data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


#自定义loader
class SemEvalDataLoader(object):
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.config = config

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        # cover array to be tensor
        sentence_feat = torch.from_numpy(np.concatenate([x[0] for x in data], axis=0)) # word id
        lexical_feat = torch.from_numpy(np.concatenate([x[1] for x in data], axis=0)) # pos for entity_1
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return (sentence_feat,lexical_feat),label

    def __get_data(self, filename, shuffle=False):
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=1,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_data('train.json', shuffle=True)

    def get_dev(self):
        return self.__get_data('test.json', shuffle=False)

    def get_test(self):
        return self.__get_data('test.json', shuffle=False)
