import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt

iris_data = load_iris()

'''
print ('Example Data:')
print (iris_data.data[:5])
print ('Example Labels: ')
print (iris_data.target[:5])

'''
x = iris_data.data
y_ = iris_data.target.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print ('Neural Network Model Summary')
print (model.summary())

# 5 and 200
model.fit(x, y, verbose=2, batch_size=5, epochs=200)

y_labelled = np.zeros(150)
for i in range(150):
    y_labelled[i] = np.argmax(y[i])

mins = np.amin(x, axis=0)
maxes = np.amax(x, axis=0)

print (mins, maxes)
rand_examples = 1000
rand_data = []

for i in range(len(mins)):
	rand_data.append(mins[i]+ (np.random.rand(rand_examples)*(maxes[i]-mins[i])))

rand_test_data = np.transpose(rand_data)


np.savetxt('random_test_data.txt', rand_test_data, fmt='%.14f')
np.savetxt('orig_test_data.txt', x, fmt='%.14f')

res_orig = model.predict_on_batch(x)
res = model.predict_on_batch(rand_test_data)

res_rand_y =np.zeros(rand_examples)
res_orig_y = np.zeros(150)
for i in range(rand_examples):    
    res_rand_y[i] = np.argmax(res[i])
    
for i in range(150):    
    res_orig_y[i] = np.argmax(res_orig[i])

np.savetxt('random_test_result.txt', res, fmt='%.14f')
np.savetxt('orig_test_result.txt', res_orig, fmt='%.14f')

X_all = np.append(x,rand_test_data,axis=0)
'''
tsne_res = TSNE(n_components=2).fit_transform(rand_test_data)
tsne_orig = TSNE(n_components=2).fit_transform(x)
'''

tsne_all = TSNE(n_components=2).fit_transform(X_all)
tsne_orig = tsne_all[0:150,:]
tsne_res = tsne_all[150:,:]

# this is the Second colourbar
plt.scatter(tsne_orig[:,0], tsne_orig[:,1], c=res_orig_y, cmap=plt.cm.get_cmap("prism", 3))
plt.colorbar(ticks=range(3))

# this is the first colourbar
plt.scatter(tsne_res[:,0], tsne_res[:,1], c=res_rand_y, cmap=plt.cm.get_cmap("jet", 3), alpha = 0.2)
plt.colorbar(ticks=range(3))
plt.title("tSNE reduced overlapped random(alpha = 0.2) and original dataset")
plt.savefig("Iris.png")
plt.show()


# # plt.subplot(122)
# # print res
# # res_r = np.array(res).reshape(3,150)
# # res_orig_r = np.array(res_orig).reshape(3,150)
# # plt.scatter(res_r[:,0], res_r[:,1],res_r[:,2], c=['r', 'g', 'b'], alpha=0.5)
# # plt.subplot(222)
# # plt.scatter(res_orig_r[:,0], res_orig[:,1],res_orig_r[:,2], c=['r', 'g', 'b'], alpha=0.5)

# np.savetxt('inp.txt', rand_data, fmt='%.14f')
# np.savetxt('res.txt', res, fmt='%.14f')


# np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
# with open('results.txt', 'w') as f:
#     f.write(np.array2string(res, separator=', '))
# results = model.evaluate(test_x, test_y)

# print 'Final test set loss: {:4f}'.format(results[0])
# print 'Final test set accuracy: {:4f}'.format(results[1])
# print results