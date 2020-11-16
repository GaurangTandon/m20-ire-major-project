import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import train_test_split

files = [ "data-018407883065180042.csv",
"data-2729217554147222.csv",
"data-4157824369063622.csv",
"data-6848830635078815.csv",
"data-1527170690458305.csv",
"data-3085637219115712.csv",
"data-43036867244493127.csv",
"data-8811109222183832.csv",
"data-15387658157043738.csv",
"data-32742980754264517.csv",
"data-4612392151534178.csv",
"data-9364298267155945.csv",
"data-22381961781535886.csv",
"data-37991020175623647.csv",
"data-5199577678267.csv",
"data-983770132587087.csv",
"data-2705507738455284.csv",
"data-41342599979795924.csv",
"data-5842712727908572.csv"
# "Dataset.csv" 
]
# files = ["Dataset.csv"]

final_list = []

for name in files:
    d = pd.read_csv(name, header=None, names=["sent", "label"], dtype={"label": float})
    for i in range(d.shape[0]):
        final_list.append(d.iloc[i].tolist())

for x in final_list:
    x.append(len(x[0]))
    x[1] = int(x[1])

df = pd.DataFrame(final_list, columns=["sent", "label", "len"])

print(df["len"].describe())
true = (df["label"] == 1).sum()
# false = df.shape[0] - true
print(true, true * 100 / df.shape[0])

# X_train, X_test, y_train, y_test = train_test_split(df["sent"], df["label"], test_size=0.2, random_state=42, shuffle=True)

df.drop(inplace=True, labels="len", axis=1)
df.to_csv("os.csv", index=False)


data = np.array(list(zip(df["sent"].to_list(), df["label"].to_list())))
seed = 42

LEN = len(data)
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

TRAIN_LEN = int(TRAIN_SIZE * LEN)
VAL_LEN = int(VAL_SIZE * LEN)
TEST_LEN = LEN - VAL_LEN - TRAIN_LEN

random.seed(seed)
random.shuffle(data)
train = data[:TRAIN_LEN, :]
val = data[TRAIN_LEN:TRAIN_LEN+VAL_LEN, :]
test = data[TRAIN_LEN+VAL_LEN:, :]

import ktrain
import pandas as pd
# import seaborn as sns
from ktrain import text
from matplotlib import pyplot as plt

print("Imports success")

"""# Dataset"""

print("Now going to run transfomers")

MAX_LEN = 100
BATCH_SIZE = 128

train_text = train[:, 0]
val_text = val[:, 0]

train_label = train[:, 1]
val_label = val[:, 1]

MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=MAX_LEN, classes=[0, 1])
trn = t.preprocess_train(train_text, train_label)
val = t.preprocess_test(val_text, val_label)
model = t.get_classifier()
# model.compile(optimizer='adam',
              # metrics=['accuracy'])
# loss=focal_loss(alpha=1, from_logits=True),
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE)

"""# Train"""

LR = 5e-5
EPOCHS = 10

print('\n')
print('\n')
print("Starting training Model now")
print('\n')
print('\n')

history = learner.autofit(LR, EPOCHS,reduce_on_plateau=2,checkpoint_folder='checkpoint_distill/')

print('\n')
print('\n')
print("Model trained now") 
print('\n')
print('\n')



predictor = ktrain.get_predictor(learner.model, preproc=t)

predictor.save('model_newdistillbert')


"""# Make Submission"""

preds = predictor.predict(test[:, 0].tolist())

print(accuracy_score(test[:, 1], preds))


