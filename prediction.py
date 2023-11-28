import pickle
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

dataset = pd.read_csv('data.csv')
diagnosis = list(dataset[:0])
diagnosis = np.delete(diagnosis, (1, -1))

print("Select a model to make a prediction: ")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
data = pickle.load(open(file_path, 'rb'))
scale = data[0]
model = data[1]
print("Fill the form to make a diagnosis:")
X_tt = []
for el in diagnosis:
    x = input(f'{el} : ')
    if el != 'id':
        X_tt.append(x)
    else:
        id = el


y_pred = model.predict(scale.transform([X_tt]))
print('...')
if y_pred.round() == 1:
    print(f'Diagnosis of {id}: Malignant')
else:
    print(f'Diagnosis of {id}: Benign')
