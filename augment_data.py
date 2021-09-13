import jsonlines
import nlpaug.augmenter.word as naw
import re
from tqdm import tqdm
import string
import random

exclude = set(string.punctuation)
base_file = 'file_v2.json1'

def build_entity_library(filename):
    # build entity library
    ent_dict = {}
    with jsonlines.open(filename) as reader:
        for obj in reader:
            for label in obj['labels']:
                start = label[0]
                end = label[1]
                ent_type = label[2]
             
                if ent_type in ent_dict:
                    ent_dict[ent_type].add(obj['text'][start:end])
                else:
                    ent_dict[ent_type] = set(obj['text'][start:end])

    return ent_dict

def synonym_replacement(sent, obj, aug_obj):
    augment_map = {}

    for label in obj['labels']:
        entity = obj['text'][label[0]:label[1]]
        aug_entity = aug_obj.augment(entity)

        augment_map[entity] = {}
        augment_map[entity]['entity'] = aug_entity
        augment_map[entity]['entity_type'] = label[2]

        sent = re.sub(r'\b{}\b'.format(entity), aug_entity, sent)
    
    return augment_map, sent

def mention_replacement(sent, obj):
    augment_map = {}

    for label in obj['labels']:
        entity = obj['text'][label[0]:label[1]]
        entity_type = label[2]
        aug_entity = random.choice(list(ent_dict[entity_type]))

        augment_map[entity] = {}
        augment_map[entity]['entity'] = aug_entity
        augment_map[entity]['entity_type'] = label[2]

        sent = re.sub(r'\b{}\b'.format(entity), aug_entity, sent)
    
    return augment_map, sent

def get_span(augment_map, sentence):
    new_span = [] 
    for ent, dt in augment_map.items():
        m = re.search(r'\b{}\b'.format(dt['entity']), sentence)
        if m is not None:
            new_span.append([m.span()[0], m.span()[1], dt['entity_type']])

    return new_span

def process_augmentation(filename, new_filename, aug_obj=None):
    new_dict = []
    with jsonlines.open(filename) as reader:
        for obj in tqdm(reader):
            new_obj = obj.copy()

            # augment the entities and change the original sentence
            sent = obj['text'][:]

            # remove punctuation
            sent = ''.join(ch for ch in sent if ch not in exclude)
            if aug_obj:
                augment_map, new_sentence = synonym_replacement(sent, obj, aug_obj)
            else:
                augment_map, new_sentence = mention_replacement(sent, obj)

            new_obj['text'] = new_sentence

            # get the new span
            new_obj['labels'] = get_span(augment_map, new_sentence)

            new_dict.append(new_obj)

    with jsonlines.open(new_filename, mode='w') as writer:
        for obj in new_dict:
            writer.write(obj)


PPDB_PATH = "C:/Users/guntu/Documents/Kuliah S3/Code & Dataset/ppdb-2.0-s-all"

ppdb_aug = naw.SynonymAug(aug_src='ppdb',model_path=PPDB_PATH)
# wordnet_aug = naw.SynonymAug(aug_src='wordnet')

process_augmentation(base_file, 'file_v2_ppdb.json', ppdb_aug)
# process_augmentation(base_file, 'file_v2_wordnet.json', wordnet_aug)

# ent_dict = build_entity_library(base_file)
# process_augmentation(base_file, 'file_v2_mention_replacement.json')











