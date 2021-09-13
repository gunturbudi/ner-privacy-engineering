import jsonlines
import random
import math


def write_to_jsonl(filename, data):
	with jsonlines.open(filename, mode='w') as writer:
		for _, line in data:
			writer.write(line)

def create_base_file(base_file):
	with jsonlines.open(base_file) as reader:
	    data = [ (random.random(), line) for line in reader ]

	data.sort()

	total_count = len(data)
	train = math.floor(0.6 * total_count)
	dev = train + math.floor(0.2 * total_count)

	train_data = data[:train]
	dev_data = data[train:dev]
	test_data = data[dev:]

	# write base file
	write_to_jsonl('shuffled_train_v2.jsonl',train_data)
	write_to_jsonl('shuffled_dev_v2.jsonl',dev_data)
	write_to_jsonl('shuffled_test_v2.jsonl',test_data)


def open_existing_train_data(base_train_file):
	with jsonlines.open(base_train_file) as reader:
	    data = [ (random.random(), line) for line in reader ]
	    return data


def create_augment_file(augment_file, train_data):
	# augment data
	augment_file = augment_file
	augment_dict = {}
	with jsonlines.open(augment_file) as reader:
	    for line in reader:
	    	augment_dict[line["id"]] = line

	# append augment data to train
	aug_train_data = []
	aug_train_data.extend(train_data)
	for dt in train_data:
		aug_train_data.append((0, augment_dict[dt[1]['id']]))

	write_to_jsonl('shuffled_train_v2_mention_replacement.jsonl',aug_train_data)

# base_file = 'file_v2.json1'
# create_base_file(base_file)
train_data = open_existing_train_data('shuffled_train_v2.jsonl')
create_augment_file('file_v2_mention_replacement.json', train_data)





