from preprocess import tokenize

from load_model import get_model

import pandas as pd
import numpy as np
import sys

def main(argv):
    emot_dic = {"anger":0, "joy":1, "sadness":2, "fear":3}

    model = get_model("model.h5")

    for s in ["anger", "joy", "sadness", "fear"]:

        df = pd.read_csv(argv[0], sep='\t', header=None, encoding='utf-8', quoting=3)
        df.columns = ['id','text','polarity','class']

        df = df[df["polarity"] == s]

        test = np.array(df['text'])
        test_type = np.array(df["polarity"])

        X = []
        for x in test:
           X.append(tokenize(x))
        X = np.array(X)

        emot = np.zeros((len(test), 4))
        for x in range(len(test_type)):
            emot[x, emot_dic[test_type[x]]] = 1

        out = model.predict([X.reshape((len(test), 50, 1)), emot.reshape((len(test), 4))])

        y_ = np.array(df["class"])
        y = np.array([int(x[0]) for x in y_])

        acc = np.count_nonzero(y == out.argmax(axis=1)) / float(out.argmax(axis=1).shape[0])

        print(s, acc)

        df["class"] = out.argmax(axis=1)

        df.to_csv("EI-oc_en_" + s + "_pred.txt", sep='\t', header=None, index=None)


if __name__ == "__main__":
    main(sys.argv[1:])
