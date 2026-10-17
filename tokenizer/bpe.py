import heapq
import regex as re
import json
from collections import Counter

PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:

	def __init__(self, max_vocab_size, special_tokens=()):
		self.max_vocab_size = max_vocab_size
		self.vocab = None
		self.merges = None
		# special tokens live outside self.vocab (they have no byte representation
		# that ordinary text can produce). Their ids come right after the merges.
		self.special_token_names = list(special_tokens)
		self.special_tokens = {}
		self.regex = re.compile(PATTERN)
		self.encode_cache = {}

	@property
	def vocab_size(self):
		return len(self.vocab) + len(self.special_tokens)

	def train(self, path_list):
		# stream the files in ~1MB chunks of whole lines instead of loading the whole corpus
		def read_chunks():
			for path in path_list:
				with open(path, "r", encoding="utf-8") as fd:
					while True:
						lines = fd.readlines(1 << 20)
						if not lines:
							break
						yield "".join(lines)

		self.train_from_iterator(read_chunks())

	def train_from_iterator(self, texts):

		# count words chunk by chunk: memory is O(unique words), not O(corpus)
		word_counts = Counter()
		nb_chars = 0
		for text in texts:
			nb_chars += len(text)
			# special tokens are cut out so they never take part in merges
			for i, part in enumerate(self._split_special(text, set(self.special_token_names))):
				if i % 2 == 0:
					word_counts.update(self.regex.findall(part))

		print(f"training corpus of {nb_chars} characters, {len(word_counts)} unique words")

		words = {word: [count, list(map(int, word.encode("utf-8")))] for word, count in word_counts.items()}

		pairs = {}
		for word in words.keys():

			word_count, byte_values = words[word]

			for i in range(len(byte_values) - 1):
				pair = (byte_values[i], byte_values[i+1])

				pairs.setdefault(pair, [0, set()])

				pairs[pair][0] += word_count
				pairs[pair][1].add(word)

		vocab = {i : bytes([i]) for i in range(256)}
		
		pair_heap = [(-count, pair, refs) for pair, (count, refs) in pairs.items()]
		heapq.heapify(pair_heap)

		merges = {}

		nb_merged_tokens = self.max_vocab_size - len(self.special_token_names)
		while len(vocab) < nb_merged_tokens and len(pair_heap) > 0:
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
							pairs.setdefault(new_pair, [0, set()])
							pairs[new_pair][0] += word_count
							pairs[new_pair][1].add(word)
							heapq.heappush(pair_heap, (-pairs[new_pair][0], new_pair, pairs[new_pair][1]))

						if i < len(byte_values) - 1:
							right_pair = (pair[1], byte_values[i+1])
							pairs[right_pair][0] -= word_count
							heapq.heappush(pair_heap, (-pairs[right_pair][0], right_pair, pairs[right_pair][1]))

							new_pair = (new_id, byte_values[i+1])
							pairs.setdefault(new_pair, [0, set()])
							pairs[new_pair][0] += word_count
							pairs[new_pair][1].add(word)
							heapq.heappush(pair_heap, (-pairs[new_pair][0], new_pair, pairs[new_pair][1]))

					i += 1
					if i >= len(byte_values) - 1:
						break

			pairs.pop(pair)

		self.vocab = vocab
		self.merges = merges
		self.special_tokens = {name: len(vocab) + i for i, name in enumerate(self.special_token_names)}
		self.encode_cache = {}

	def _split_special(self, text, allowed):
		# -> [ordinary, special, ordinary, special, ..., ordinary]
		# (even indices are ordinary text, odd indices are allowed special tokens)
		if not allowed:
			return [text]
		special_pattern = "(" + "|".join(re.escape(name) for name in sorted(allowed, key=len, reverse=True)) + ")"
		return re.split(special_pattern, text)

	def encode(self, text, allowed_special="all"):
		# allowed_special: "all", "none", or a set of special token strings.
		# anything not allowed is encoded as plain text (byte level), which is
		# what we want for web documents that happen to contain "<|endoftext|>"
		if allowed_special == "all":
			allowed = set(self.special_tokens)
		elif allowed_special == "none":
			allowed = set()
		else:
			allowed = set(allowed_special)

		tokens = []
		for i, part in enumerate(self._split_special(text, allowed)):
			if i % 2 == 1:
				tokens.append(self.special_tokens[part])
			else:
				tokens.extend(self.encode_ordinary(part))
		return tokens

	def encode_ordinary(self, text):

		tokens = []

		for word in self.regex.findall(text):
			# TODO(step 5, yours): memoize word -> token ids in self.encode_cache.
			# Words repeat a lot in natural text, so most calls become a dict lookup.
			# tests/test_bpe_cache.py tells you when it's right.
			if word in self.encode_cache:
				word_bytes = self.encode_cache[word]
			else:
				word_bytes = self._encode_word(word)
				self.encode_cache[word] = word_bytes
			tokens.extend(word_bytes)

		return tokens

	def _encode_word(self, word):

		byte_values = list(map(int, word.encode("utf-8")))

		while len(byte_values) > 1:
			pairs = {(byte_values[i], byte_values[i+1]) for i in range(len(byte_values) - 1)}

			pair = min(pairs, key= lambda x : self.merges.get(x, float("inf")))

			if pair not in self.merges.keys():
				break

			i = 0
			while True:
				if byte_values[i] == pair[0] and byte_values[i+1] == pair[1]:
					byte_values.pop(i+1)
					byte_values[i] = self.merges[pair]

				i += 1
				if i >= len(byte_values) - 1:
					break

		return byte_values

	def decode(self, tokens, errors="replace"):
		# errors="replace": a model can sample byte sequences that aren't valid
		# utf-8 on their own (half of a multi-byte character), don't crash on them
		id_to_special = {v: k.encode("utf-8") for k, v in self.special_tokens.items()}
		return b"".join(id_to_special[t] if t in id_to_special else self.vocab[t] for t in tokens).decode("utf-8", errors=errors)

	def dump(self, path):
		data = {
			"max_vocab_size": self.max_vocab_size,
			"vocab": {str(k): list(v) for k, v in self.vocab.items()},
			"merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
			"special_tokens": self.special_tokens,
		}
		with open(path, "w", encoding="utf-8") as fd:
			json.dump(data, fd)

	def load(self, path):
		with open(path, "r", encoding="utf-8") as fd:
			data = json.load(fd)
		self.max_vocab_size = data["max_vocab_size"]
		self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
		self.merges = {
			tuple(int(x) for x in k.split(",")): v
			for k, v in data["merges"].items()
		}
		self.special_tokens = data.get("special_tokens", {})
		self.special_token_names = list(self.special_tokens)
		self.encode_cache = {}
