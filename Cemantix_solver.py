import requests
from gensim.models import KeyedVectors
import time
import random
# 

def tryWordOnCemantix(word):
	url = 'https://cemantix.herokuapp.com/score'
	data={
	    "word": word,
	}

	response = requests.post(url, data=data).json()
	try:
		score = response['score']
	except:
		score = -1000

	return score


def main(model_name, iter_max=100, similarity_threshold=0.001):

	print("Loading word2vec model")
	start_time = time.time()
	word2vec_model = KeyedVectors.load_word2vec_format(f"{model_name}.bin", binary=True, unicode_errors="ignore")

	print(f"model load in {(time.time() - start_time):.2f} s")

	# try a radom word to start
	word_to_try = random.sample(word2vec_model.index_to_key, 1)

	candidate_list = [word_to_try] #]
	dict_results = dict()
	try_list = []
	iter_i = 1
	score = 0
	while (score != 1) & (iter_i <= iter_max):

		if candidate_list:
			word_to_try = candidate_list.pop(0)
		else:
			# try a random selected word
			word_to_try = random.sample(word2vec_model.index_to_key, 1)[0]

		score = tryWordOnCemantix(word=word_to_try)
		try_list.append((word_to_try, score))

		if score == 1:
			print(f"Victory the word {word_to_try} is the guess!!! ")
			print(f"count iteration:: {iter_i}")
			print(f"count word try:: {len(try_list)}")

			return True

		if score == -1000:
			print(f"unknown word: {word_to_try}, trying an other guess")
			continue
		
		print(f"Iter-{iter_i}: try word='{word_to_try}'... Similarity with the guess={score}")
		
		list_similar_word = word2vec_model.most_similar(word_to_try, topn=200000)
		list_similar_word = [
			(word, sim, abs(round(score, 4) - round(sim, 4))) 
			for word, sim in list_similar_word 
				if abs(round(score, 4) - round(sim, 4)) <= similarity_threshold
		]
		list_similar_word = sorted(list_similar_word, key=lambda x: x[2])
		print(f"{len(list_similar_word)} candidates")
		dict_results[word_to_try] = list_similar_word

		# find next word to try
		# test best word
		most_similar_word = sorted(try_list, key=lambda x: -x[1])[0]
		try_word_list = [word for word, _ in try_list]
		candidate_list = [word for word, _, _ in dict_results[most_similar_word[0]] if word not in try_word_list]

		print(f"most_similar_word: {most_similar_word} --> {candidate_list[:10]}")

		iter_i += 1

	print(f"maximum iteration reach")
	return False

if __name__ == "__main__":

	# Get word2vec model
	model_str = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100 (3)" # "frWac_non_lem_no_postag_no_phrase_200_skip_cut100"
	main(model_name=model_str, iter_max = 100, similarity_threshold=0.01)

