from preprocess import tokenize
from load_model import get_model

import numpy as np
import sys

def main(argv):
    emot_dic = {"anger":0, "joy":1, "sadness":2, "fear":3}

    x = tokenize(argv[0])

    emot = np.array([0, 0, 0, 0])
    emot[emot_dic[argv[1]]] = 1

    model = get_model("model.h5")

    out = model.predict([x.reshape((1, 50, 1)), emot.reshape((1, 4))])[0]

    print(argv[1], ":",  np.argmax(out))

if __name__ == "__main__":
    main(sys.argv[1:])
