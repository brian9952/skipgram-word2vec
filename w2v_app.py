import numpy as np
import pandas as pd
import json

def import_model(file):
    with open(file) as json_file:
        data = json.load(json_file)

    return data

def similarity(a, b):
    dot = np.dot(a, b)

    summation = 0
    for i in range(len(a)):
        mul = a[i] ** 2
        summation = summation + mul

    magn_a = np.sqrt(summation)

    summation = 0
    for i in range(len(b)):
        mul = b[i] ** 2
        summation = summation + mul

    magn_b = np.sqrt(summation)

    similarity = dot / (magn_a * magn_b)
    return similarity

class w2v_app:
    def __init__(self, vocab, w1, w2):
        self.w1 = w1
        self.w2 = w2
        self.vocab = vocab
        self.words = list(vocab.keys())

    def getVector(self, string):
        label_dict = self.vocab

        if string not in self.vocab:
            print("word not found! please try another word")
            return
        
        arr_index = label_dict[string]
        str_vector = self.w1[arr_index]

        return np.array(str_vector)

    def print_similar(self, sim_dict):
        sort_dict = dict(sorted(sim_dict.items(), key=lambda item:item[1], reverse = True))
        top = 5

        for i in range(top):
            print("Word = {}, Similarity = {}".format(
                list(sort_dict.keys())[i],
                list(sort_dict.values())[i]
            ))


    def find_similar(self, word):
        all_words = self.words
        similarity_dict = dict()

        for i in range(len(all_words)):

            if all_words[i] == word:
                continue

            word_vec = self.getVector(word)

            other_vec = self.getVector(all_words[i])

            sim = similarity(word_vec, other_vec)

            similarity_dict[all_words[i]] = sim

        print("Top similar to = {}".format(word))
        self.print_similar(similarity_dict)


if __name__ == "__main__":

    # import trained model
    data = import_model("model.json")

    model = w2v_app(data["vocab"], data["w1"], data["w2"])

    while True:
        word = input("Input a word \n")
        model.find_similar(word)
