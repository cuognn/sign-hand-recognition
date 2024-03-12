import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA


data_dict = pickle.load(open('./data.pickle', 'rb'))
# import pdb; pdb.set_trace()

# Assuming data_dict['data'] is a list of sequences with varying lengths
max_length = max(len(seq) for seq in data_dict['data'])

# Pad sequences to make them uniform in length
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

# Convert to NumPy array
data = np.asarray(padded_data)
# # Apply PCA to reduce the dimensionality from 84 to 42
# pca = PCA(n_components=42)
# data = pca.fit_transform(data)
# import pdb; pdb.set_trace()

# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

