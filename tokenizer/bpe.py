
import heapq
import regex as re

class BPETokenizer:

	def __init__(self, max_vocab_size):
		self.max_vocab_size = max_vocab_size
		self.vocab = None
		self.merges = None

	def train(self, path_list):

		texts = []
		for path in path_list:
			with open(path, "r", encoding="utf-8") as fd:
				texts.append(fd.read())
		corpus = "\n\n".join(texts)

		pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
		regex = re.compile(pattern)

		words_list = regex.findall(corpus)

		words = {}
		for word in words_list:
			words.setdefault(word, [0, list(map(int, word.encode("utf-8")))])
			words[word][0] += 1

		pairs = {}
		for word in words.keys():

			word_count, byte_values = words[word]

			for i in range(len(byte_values) - 1):
				pair = (byte_values[i], byte_values[i+1])

				pairs.setdefault(pair, [0, []])

				pairs[pair][0] += word_count
				pairs[pair][1].append(word)

		vocab = {i : bytes([i]) for i in range(256)}
		
		pair_heap = [(-count, pair, refs) for pair, (count, refs) in pairs.items()]
		heapq.heapify(pair_heap)

		merges = {}

		while len(vocab) < self.max_vocab_size and len(pair_heap) > 0:
			count, pair, refs = heapq.heappop(pair_heap)
			if pair not in pairs.keys() or pair in pairs.keys() and -count != pairs[pair][0]:
				continue

			new_id = len(vocab)
			vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

			merges[pair] = new_id
			
			for word in refs:
				word_count, byte_values = words[word]

				i = 0
				while True:
					if byte_values[i] == pair[0] and byte_values[i+1] == pair[1]:
						byte_values.pop(i+1)
						byte_values[i] = new_id

						if i > 0:
							left_pair = (byte_values[i-1], pair[0])
							pairs[left_pair][0] -= word_count
							heapq.heappush(pair_heap, (-pairs[left_pair][0], left_pair, pairs[left_pair][1]))

							new_pair = (byte_values[i-1], new_id)
							pairs.setdefault(new_pair, [0, []])
							pairs[new_pair][0] += word_count
							pairs[new_pair][1].append(word)
							heapq.heappush(pair_heap, (-pairs[new_pair][0], new_pair, pairs[new_pair][1]))

						if i < len(byte_values) - 1:
							right_pair = (pair[1], byte_values[i+1])
							pairs[right_pair][0] -= word_count
							heapq.heappush(pair_heap, (-pairs[right_pair][0], right_pair, pairs[right_pair][1]))

							new_pair = (new_id, byte_values[i+1])
							pairs.setdefault(new_pair, [0, []])
							pairs[new_pair][0] += word_count
							pairs[new_pair][1].append(word)
							heapq.heappush(pair_heap, (-pairs[new_pair][0], new_pair, pairs[new_pair][1]))

					i += 1
					if i >= len(byte_values) - 1:
						break

			pairs.pop(pair)

		print(vocab)
		self.vocab = vocab
		self.merges = merges
	pass
