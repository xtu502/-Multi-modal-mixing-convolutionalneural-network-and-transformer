# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:23:49 2023

@author: 47874
"""

from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report
from glob import glob
import numpy as np
import sklearn as sk
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')   #去除Xmanager软件来处理X11
import tensorflow as tf
import tensorflow

os.getcwd()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



#Read data
dat = pd.read_csv('/home/junde/51_Alzheimer_disease/01_dataset/AD/ADNI_Training_Q3_APOE_CollectionADNI1Complete 1Yr 1.5T_July22.2014.csv')
dat = dat.dropna()
dat.isnull().sum().sum()

dat.head(3)

X = dat
Y = dat['DX.bl']
del dat


all_xray_df = X

all_image_paths = {os.path.basename(x): x for x in 
                    glob(os.path.join('/home/junde/51_Alzheimer_disease/01_dataset/', 'Total', '*.jpg'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

PD_Img =all_xray_df['Subject'].astype('str') + '.jpg'
all_xray_df['path'] = PD_Img.map(all_image_paths.get)

all_xray_df = all_xray_df.loc[all_xray_df['path'].notnull()]
all_xray_df.sample(3)


ID_lbl = []
images = []
## Read the Images ##
for i in range(1, 627):
    img = cv2.imread(all_xray_df['path'].iloc[i])
    lbl = all_xray_df[['Image.Data.ID', 'Subject', 'DX.bl']].iloc[i]
    gray = img
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = gray/255
    gray = cv2.resize(gray,(200, 200))
    images.append(gray)
    ID_lbl.append(lbl)  # MCI
    
# Sample Images
plt.figure(figsize = (15,15))
for i in range(10):
    plt.subplot(4, 5, i + 1)
    plt.imshow(images[1 + i*3])
plt.show()

ID_lbl = pd.DataFrame(ID_lbl)
pattern = {'AD':0, 'CN':1, 'LMCI':2}
ID_lbl['DX.bl'] = [pattern[x] if x in pattern else x for x in ID_lbl['DX.bl']]


import numpy as np
train_feature = np.array(images)
ID_lbl = np.array(ID_lbl)

## Display Array Shape ##
print(f"image dataset shape = {train_feature.shape}")


from sklearn.model_selection import train_test_split
train_features, test_features, train_idlbl, test_idlbl = train_test_split(train_feature,ID_lbl,test_size=0.12)
train_id = train_idlbl[:,0:2]
test_id = test_idlbl[:,0:2]
train_target = train_idlbl[:,-1] 
test_target = test_idlbl[:,-1]
print(f"train_features shape = {train_features.shape}")
print(f"test_features shape = {test_features.shape}")
print(f"train_target shape = {train_target.shape}")
print(f"test_target shape = {test_target.shape}")



# MODEL ARCHITECTURE
import keras
import tensorflow
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
# ONE NOT ENCODING
train_target = to_categorical(train_target)
test_target = to_categorical(test_target)

'''
def get_CRXMDL(model_input):
    model=Sequential()
    model.add(Conv2D(25, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu', input_shape = (200, 200, 3)))
    model.add(Conv2D(75, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(3,activation='softmax'))    
    model.compile(Adam(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model   
CRXMDL = get_CRXMDL(train_features.shape[1:])    
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 



data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
#        layers.Normalization(),
#        layers.Resizing(image_size, image_size),
#        layers.RandomFlip("horizontal"),
#        layers.RandomRotation(factor=0.02),
#        layers.RandomZoom(
#            height_factor=0.2, width_factor=0.2
#        ),
#    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(train_features)



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def get_CRXMDL(model_input):
    input_shape = (200, 200, 3)
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(3, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) 
    return model
CRXMDL = get_CRXMDL(train_features.shape[1:])  



# --------------Tabular data Modeling-----------------------------
#Remove unnecessary columns (features), remove first 9 columns and 'Dx codes for submission'
remove_columns = list(X.columns)[4:9]
remove_columns.append('Dx Codes for Submission')
print('Removing columns:', remove_columns)

X = X.drop(remove_columns, axis=1)
X = X.drop(['directory.id','RID','path'], axis=1)

features = list(X.iloc[:,2:].columns)

numerical_vars = ['AGE', 'MMSE', 'PTEDUCAT']
cat_vars = list(set(features) - set(numerical_vars))
print('Categorical variable distributions:\n')
for var in cat_vars:
    print('\nDistribution of', var)
    print(X[var].value_counts())


from matplotlib import pyplot as plt
# %matplotlib inline

print('Numerical Var Distributions:\n')

#for each categorical var, convert to 1-hot encoding
for var in cat_vars:
    print('Converting', var, 'to 1-hot encoding')    
    #get 1-hot and replace original column with the >= 2 categories as columns
    one_hot_df = pd.get_dummies(X[var])
    X = pd.concat([X, one_hot_df], axis=1)
    X = X.drop(var, axis=1)
X.head(4)


def normalize(X):
    #Convert to numpy array
    X = np.array(X)
    sanity_check = 0
    #Normalize numerical variables to speed up convergence
    for i in range(3):
        mean = np.mean(X[:, i])
        sd = np.std(X[:, i])
        print('\nNormalizing', numerical_vars[i], 'with mean=', format(mean, '.2f'), 'and sd=', format(sd, '.2f'))
        X[:, i] = (X[:, i] - mean) / sd
        sanity_check += np.mean(X[:, i])
    print('\nSanity Check. Sum of all the means should be near 0:', sanity_check)
    return X



train_id = pd.DataFrame(train_id)
train_id.columns = ['Image.Data.ID', 'Subject']
test_id = pd.DataFrame(test_id) 
test_id.columns = ['Image.Data.ID', 'Subject']
X_train = pd.merge(train_id,X,left_on=['Image.Data.ID', 'Subject'],right_on=['Image.Data.ID', 'Subject'],how="left")
X_test = pd.merge(test_id,X,left_on=['Image.Data.ID', 'Subject'],right_on=['Image.Data.ID', 'Subject'],how="left")
X_train = X_train.iloc[:,2:]
X_test = X_test.iloc[:,2:]


from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras import backend as K
from keras.models import Model
from keras.layers.core import Flatten
input_img = Input(shape=(22, 1))  # 一维卷积输入层，神经元个数为400（特征数）
def get_TSMDL(model_input):
    x1 = Conv1D(32, 3, activation='relu', padding='same')(input_img)   #卷积层，32个核，宽度为5，激活函数为relu
    x1 = Conv1D(32, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)   #  规范层，加速模型收敛，防止过拟合
    x1 = MaxPooling1D(2, padding='same')(x1)   # 池化层
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2, padding='same')(x1)
    x1 = Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2, padding='same')(x1)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    encoded = MaxPooling1D(2, padding='same')(x1)  # 全连接层
    encoded = Flatten()(encoded)
    decoded = Dense(3, activation='softmax')(encoded)   # softmax激活函数，输出层
    # decoded = Dense(1, activation='linear')(encoded)   # Regression问题  linear激活函数，输出层
    autoencoder = Model(input_img, decoded)   #编码
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) 
    return autoencoder
TSMDL = get_TSMDL(input_img)




combinedInput = keras.layers.concatenate([CRXMDL.output, TSMDL.output])
x = Dense(100, activation="relu", kernel_initializer='he_uniform')(combinedInput)
x = Dense(3, activation="softmax")(x)
# model = Model(inputs=[CRXMDL.input, TSMDL.input], outputs=x)
model = Model(inputs=[CRXMDL.input, TSMDL.input], outputs=x)
# model = Model(inputs=[tf.convert_to_tensor(CRXMDL.input), TSMDL.input], outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(
	[train_features, X_train], train_target,
	validation_data=([test_features, X_test], test_target),
	epochs=100, batch_size=256)



# 模型预测      sparse_categorical_crossentropy
pre_trn = model.predict([train_features, X_train])
pre_trn = np.argmax(pre_trn, axis=1)   # 训练集预测结果格式转换，形成向量
trn_LBL = np.argmax(train_target, axis=1) 

preds = model.predict([test_features, X_test])
predict = np.argmax(preds, axis=1)   # 训练集预测结果格式转换，形成向量
tst_LBL = np.argmax(test_target, axis=1) 

#----统计指标-1，训练集---------------------------------------------------------------------
trn_acc=accuracy_score(trn_LBL, pre_trn)
print('trn_acc='+str(trn_acc))

trn_recall=recall_score(trn_LBL, pre_trn, average='macro')  #'macro'
print('trn_recall='+str(trn_recall))

trn_f1_score = f1_score(trn_LBL, pre_trn, average='weighted')  
print('trn_f1_score='+str(trn_f1_score))

#----统计指标-2，测试集---------------------------------------------------------------------
tst_acc=accuracy_score(tst_LBL, predict)
print('tst_acc='+str(tst_acc))

tst_recall=recall_score(tst_LBL, predict, average='macro')  #'macro'
print('tst_recall='+str(tst_recall))

tst_f1_score = f1_score(tst_LBL, predict, average='weighted')  
print('tst_f1_score='+str(tst_f1_score))







