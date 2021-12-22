import numpy as np
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import json

# import all data
def importData(dirs):
    empty = {
        "ArticleTitle": [],
        "Question": [],
        "Answer": [],
        "DifficultyFromQuestioner": [],
        "DifficultyFromAnswerer": [],
        "ArticleFile": []
    }

    df = pd.DataFrame(empty)

    for i in range(len(dirs)):
        data = pd.read_csv(dirs[i], delimiter = "\t", encoding = "ISO-8859-1")
        df = df.append(data)

    df = df[["Question", "Answer"]]
    df = df.dropna()
    df = df.drop_duplicates(subset = ["Question"])
    return df

def removePuncFunc(text):
    result = "".join([i for i in text if i not in string.punctuation])
    return result

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def BOSEOS(sentence_arr):
    bos = "<BOS>"
    eos = "<EOS>"

    sentence_arr.insert(0, bos)
    sentence_arr.insert(len(sentence_arr) + 1, eos)

    return sentence_arr

class preprocess:
    def __init__(self, dataset):
        self.dataset = dataset.values

    def removePunc(self):
        new_dataset = []

        for i in range(len(self.dataset)):
            punc_free_dataset = removePuncFunc(self.dataset[i])
            new_dataset.append(punc_free_dataset)

        self.dataset = new_dataset

    def lowercase(self):
        new_dataset = []

        for i in range(len(self.dataset)):
            dataset_lower = self.dataset[i].lower()
            new_dataset.append(dataset_lower)

        self.dataset = new_dataset

    def tokenize(self):
        new_dataset = []

        for i in range(len(self.dataset)):
            text_token = nltk.tokenize.word_tokenize(self.dataset[i])
            new_dataset.append(text_token)

        self.dataset = new_dataset

    def addBOSEOS(self):
        new_dataset = []

        for i in range(len(self.dataset)):
            boseos = BOSEOS(self.dataset[i])
            new_dataset.append(boseos)

        self.dataset = new_dataset

class encoding:
    def __init__(self, dataset):
        self.label_dict = dict()
        self.dataset = dataset
        self.label_encoded = []
        self.onehot_encoded = []
    
    def createVocab(self):
        temp = []

        dataset_flatten = [item for sublist in self.dataset for item in sublist]
 
        for i in range(len(dataset_flatten)):
            
            if dataset_flatten[i] not in temp:
                temp.append(dataset_flatten[i])
        
        temp_dict = dict()
        for i in range(len(temp)):
            temp_dict[temp[i]] = i

        self.label_dict = temp_dict

    def labelEncoding(self):
        self.createVocab()

        new_dataset = []

        for i in range(len(self.dataset)):
            inner_arr = []
            
            for j in range(len(self.dataset[i])):
                label = self.label_dict[self.dataset[i][j]]
                inner_arr.append(label)

            new_dataset.append(inner_arr)

        self.label_encoded = new_dataset

    def oneHotEncoding(self, max_vocab):
        encoded = []

        for i in range(len(self.label_encoded)):
            inner_arr = []

            for j in range(len(self.label_encoded[i])):
                onehot = [0] * max_vocab
                onehot[self.label_encoded[i][j]] = 1
                inner_arr.append(onehot)

            encoded.append(inner_arr)
        
        self.onehot_encoded = encoded

class word2vec:
    def __init__(self, dataset, window_size):
        self.embedding_size = 100
        self.w1 = np.random.uniform(low = -1, high = 1, size = (len(dataset[0][0]), self.embedding_size))
        self.w2 = np.random.uniform(low = -1, high = 1, size = (self.embedding_size, len(dataset[0][0])))
        self.dataset = dataset
        self.window_size = window_size
    
    def getContext(self, pos, dataset_row, low, high, window_size):
        context_arr = []
        temp = 0
        
        # windowing backwards   
        for i in range(pos, low, -1):

            if i == low or i == (pos - window_size):
                break

            context_arr.append(dataset_row[i-1])

        temp = 0

        # windowing forwards
        for i in range(pos, high):

            if i == (high - 1) or i == (pos + window_size):
                break

            context_arr.append(dataset_row[i+1])

        return context_arr

    def feedForward(self, target_arr, context_arr):
        sum_diff = 0
        target_arr = np.array(target_arr)
        context_arr = np.array(context_arr)

        hidden = np.dot(target_arr, self.w1)
        output = np.dot(hidden, self.w2)
        y_pred = softmax(output)

        # calculate sum of diff
        for i in range(len(context_arr)):
            diff = y_pred - context_arr[i]
            sum_diff = sum_diff + diff

        # calculate error
        error = - np.max(output) + len(context_arr) * np.log(np.sum(np.exp(output)))
        
        return sum_diff, hidden, error
 
    def backprop(self, target_arr, sum_diff, hidden, learning_rate):
        target_arr = np.array(target_arr)

        D_w2 = np.outer(hidden, sum_diff)
        dot = np.dot(self.w2, sum_diff)
        D_w1 = np.outer(target_arr, dot)
        
        # update weights
        self.w1 = self.w1 - (learning_rate * D_w1)
        self.w2 = self.w2 - (learning_rate * D_w2)

    def train(self, epoch, learning_rate):
#        error_arr = []

        for i in range(epoch): # i for iteration of epoch
            error_arr = []

            for j in range(len(dataset)): # j for iteration of each row of dataset
                window_size = self.window_size
                
                if len(dataset[j]) < window_size:
                    continue
                
                inner_arr = []
                low = 0
                high = len(dataset[j])
                print("Training progress = {}%".format(round((j / len(dataset) * 100), 2)), end = "\r")

                for k in range(len(dataset[j])): # k for iteration of each column of dataset
                    temp = 0

                    context_arr = self.getContext(k, dataset[j], low, high, window_size)
                    
                    # feed forward
                    sum_diff, hidden, error = self.feedForward(dataset[j][k], context_arr)
                    inner_arr.append(error)

                    # backprop
                    self.backprop(dataset[j][k], sum_diff, hidden, learning_rate)
                        
                mean_error = np.mean(np.array(inner_arr))
            
            error_arr.append(mean_error)
            print("epoch = {}, mean_error = {}".format(i + 1, np.array(np.mean(error_arr))))

        # show error plot
        #plt.plot(error_arr)
        #plt.show()

def save_model(w1, w2, vocab):
    model_dict = {
        "w1": w1,
        "w2": w2,
        "vocab": vocab
    }

    with open("model.json", "w") as outfile:
        json.dump(model_dict, outfile)


if __name__ == "__main__":

    dirs = [
        "Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt",
        "Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt",
        "Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt"
    ]

    # import dataset
    df = importData(dirs)

    # create dataset for embedding
    dataset = df["Question"].append(df["Answer"], ignore_index = True)
    
    # create object for preprocessing steps
    new_dataset = preprocess(dataset)
    
    # dataset preprocessing steps
    new_dataset.removePunc()
    new_dataset.lowercase()
    new_dataset.tokenize()
    new_dataset.addBOSEOS()
    dataset = new_dataset.dataset

    # dataset encoding
    new_dataset = encoding(dataset)
    new_dataset.labelEncoding()
    new_dataset.oneHotEncoding(5800)
    dataset = new_dataset.onehot_encoded

    # word2vec train
    window_size = 2
    learning_rate = 0.005
    model = word2vec(dataset, window_size)
    model.train(10, learning_rate)

    # save the vocab and model into json files
    w1 = model.w1.tolist()
    w2 = model.w2.tolist()
    vocab = new_dataset.label_dict
    save_model(w1, w2, vocab)

