# :hospital: COVID19 Diagnosis with CT Images :computer:

> [Kaggle CT COVID19 Dataset page](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/4f22df0f-17e5-4ab2-b416-c62d4f2a16bb)

## Coronavirus Disease 2019 (COVID-19)

Coronavirus disease 2019 (COVID-19) is defined as illness caused by a novel coronavirus now called severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2; formerly called 2019-nCoV), which was first identified amid an outbreak of respiratory illness cases in Wuhan City, Hubei Province, China. It was initially reported to the WHO on December 31, 2019. On January 30, 2020, the WHO declared the COVID-19 outbreak a global health emergency. On March 11, 2020, the WHO declared COVID-19 a global pandemic, its first such designation since declaring H1N1 influenza a pandemic in 2009.

## What is ResNet 50 model?

ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals, ResNet is the short name for residual Network.

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/0e8ac762-5912-447d-80f4-968f099abfa4)


## :clipboard: Mini Project (2021/04/01 ~ 2021/11/28) :date:

> :family: 팀명: Are you still there?
- 김대영
- [지우근](https://github.com/UGeunJi)

---

## :scroll: 프로젝트에 대한 전반적인 설명

### 주제 : ResNet50을 전이학습한 모델을 이용하여 CT 이미지를 통해 코로나를 진단

#### 1. 데이터 준비 과정 

```
(1) 데이터 로드
(2) 데이터 정보 시각화
(3) 데이터 전처리 (정규화)
(4) train, validation set 구분
```

#### 2. 모델 생성

```
(1) Resnet50 Transfer Learning
(2) 손실함수, 옵티마이저, 학습률, 학습 스케쥴러 설정
```

#### 3. 모델 훈련 및 성능 검증

```
(1) 경진대회 아닌 경우 : 평가 (정답이 있음)
(2) 경진대회인 경우 : 예측 및 제출(캐글에서 평가받을 수 있음)
```

```
(1) 예측 결과 확인 (Confusion Matrix, Plot)
(2) Test Image 시험
```

---

# :computer: 실행 코드

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
```

```
disease_types=['COVID', 'non-COVID']
data_dir = '../input/sarscov2-ctscan-dataset'
train_dir = os.path.join(data_dir)
```

```
train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])

train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.head()
```

| |	File |	DiseaseID	Disease | Type |
| --- | --- | --- | --- |
| 0	| COVID/Covid (230).png	| 0	| COVID |
| 1	| COVID/Covid (1195).png |	0	| COVID |
| 2	| COVID/Covid (182).png	| 0	| COVID |
| 3	| COVID/Covid (817).png	| 0	| COVID | 
| 4	| COVID/Covid (631).png |	0	| COVID |

```        
SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()
```

```
import pandas_profiling as pp
pp.ProfileReport(train)
```

```
Details about Dataset
```

```
plt.hist(train['DiseaseID'])
plt.title('Frequency Histogram of Species')
plt.figure(figsize=(12, 12))
plt.show()
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/fae35bfb-33f6-445b-9b2a-a00b948552fb)

```
def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('COVID', 3, 3)
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/b961cc95-fcf1-4dcf-b44e-5437cd07c851)

```
def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('non-COVID', 3, 3)
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/2419d5b4-6c47-4f0b-8c9f-e95326567b2d)

```
IMAGE_SIZE = 64
def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
```

```
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X_Train = X_train / 255.
print('Train Shape: {}'.format(X_Train.shape))
```

```
2481it [00:30, 82.16it/s]
Train Shape: (2481, 64, 64, 3)
```

```
Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=2)
print(Y_train.shape)
```

```
(2481, 2)
```

```
BATCH_SIZE = 64

# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)
```

```
print(f'X_train:',X_train.shape)
print(f'X_val:',X_val.shape)
print(f'Y_train:',Y_train.shape)
print(f'Y_val:',Y_val.shape)
```

```
X_train: (1984, 64, 64, 3)
X_val: (497, 64, 64, 3)
Y_train: (1984, 2)
Y_val: (497, 2)
```

```
fig, ax = plt.subplots(1, 3, figsize=(15, 15))
for i in range(3):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])
    ax[i].set_title(disease_types[np.argmax(Y_train[i])])
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/dee2ce6e-e97d-4d14-b789-f80ee50ac67d)

```
EPOCHS = 100
SIZE=64
N_ch=3
```

```
def build_resnet50():
    resnet50 = ResNet50(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = resnet50(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(2,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model
```

```
model = build_resnet50()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.70, patience=5, verbose=1, min_lr=1e-4)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(X_train)
```

```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

94773248/94765736 [==============================] - 0s 0us/step
Model: "functional_1"
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/a27a70aa-8cc6-43cf-a9cb-895e02319477)

```
from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='convnet.png', show_shapes=True,show_layer_names=True)
Image(filename='convnet.png') 
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/c379850d-55d6-4702-b716-c9214f61df70)

```
# Fits the model on batches with real-time data augmentation
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))
```

Epoch 1/100
31/31 [==============================] - ETA: 0s - loss: 1.1960 - accuracy: 0.5383
Epoch 00001: val_loss improved from inf to 4.50768, saving model to model.h5
31/31 [==============================] - 5s 176ms/step - loss: 1.1960 - accuracy: 0.5383 - val_loss: 4.5077 - val_accuracy: 0.4487

Epoch 2/100
31/31 [==============================] - ETA: 0s - loss: 1.0704 - accuracy: 0.5827
Epoch 00002: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 96ms/step - loss: 1.0704 - accuracy: 0.5827 - val_loss: 9.4418 - val_accuracy: 0.4487

Epoch 3/100
31/31 [==============================] - ETA: 0s - loss: 0.9108 - accuracy: 0.6195
Epoch 00003: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 98ms/step - loss: 0.9108 - accuracy: 0.6195 - val_loss: 5.3648 - val_accuracy: 0.4487

Epoch 4/100
31/31 [==============================] - ETA: 0s - loss: 0.8254 - accuracy: 0.6517
Epoch 00004: val_loss did not improve from 4.50768
31/31 [==============================] - 4s 129ms/step - loss: 0.8254 - accuracy: 0.6517 - val_loss: 18.8762 - val_accuracy: 0.4487

Epoch 5/100
31/31 [==============================] - ETA: 0s - loss: 0.7385 - accuracy: 0.6825
Epoch 00005: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 95ms/step - loss: 0.7385 - accuracy: 0.6825 - val_loss: 16.6468 - val_accuracy: 0.4487

Epoch 6/100
31/31 [==============================] - ETA: 0s - loss: 0.6954 - accuracy: 0.6749
Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.002100000018253922.

Epoch 00006: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 97ms/step - loss: 0.6954 - accuracy: 0.6749 - val_loss: 14.8384 - val_accuracy: 0.4487

Epoch 7/100
31/31 [==============================] - ETA: 0s - loss: 0.5987 - accuracy: 0.7374
Epoch 00007: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 106ms/step - loss: 0.5987 - accuracy: 0.7374 - val_loss: 6.9381 - val_accuracy: 0.4487

Epoch 8/100
31/31 [==============================] - ETA: 0s - loss: 0.5406 - accuracy: 0.7510
Epoch 00008: val_loss did not improve from 4.50768
31/31 [==============================] - 3s 97ms/step - loss: 0.5406 - accuracy: 0.7510 - val_loss: 20.0613 - val_accuracy: 0.4487

Epoch 9/100
31/31 [==============================] - ETA: 0s - loss: 0.5255 - accuracy: 0.7641
Epoch 00009: val_loss improved from 4.50768 to 1.45244, saving model to model.h5
31/31 [==============================] - 4s 135ms/step - loss: 0.5255 - accuracy: 0.7641 - val_loss: 1.4524 - val_accuracy: 0.5513

Epoch 10/100
31/31 [==============================] - ETA: 0s - loss: 0.5246 - accuracy: 0.7671
Epoch 00010: val_loss did not improve from 1.45244
31/31 [==============================] - 3s 104ms/step - loss: 0.5246 - accuracy: 0.7671 - val_loss: 9.2468 - val_accuracy: 0.4487

Epoch 11/100
31/31 [==============================] - ETA: 0s - loss: 0.4565 - accuracy: 0.7994
Epoch 00011: val_loss did not improve from 1.45244
31/31 [==============================] - 3s 98ms/step - loss: 0.4565 - accuracy: 0.7994 - val_loss: 18.6173 - val_accuracy: 0.4487

Epoch 12/100
31/31 [==============================] - ETA: 0s - loss: 0.4693 - accuracy: 0.8014
Epoch 00012: val_loss did not improve from 1.45244
31/31 [==============================] - 3s 95ms/step - loss: 0.4693 - accuracy: 0.8014 - val_loss: 23.2059 - val_accuracy: 0.4487

Epoch 13/100
31/31 [==============================] - ETA: 0s - loss: 0.4297 - accuracy: 0.8165
Epoch 00013: val_loss did not improve from 1.45244
31/31 [==============================] - 3s 97ms/step - loss: 0.4297 - accuracy: 0.8165 - val_loss: 2.5663 - val_accuracy: 0.4507

Epoch 14/100
31/31 [==============================] - ETA: 0s - loss: 0.4311 - accuracy: 0.8065
Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.0014699999475851653.

Epoch 00014: val_loss did not improve from 1.45244
31/31 [==============================] - 4s 119ms/step - loss: 0.4311 - accuracy: 0.8065 - val_loss: 6.5573 - val_accuracy: 0.4487

Epoch 15/100
31/31 [==============================] - ETA: 0s - loss: 0.4150 - accuracy: 0.8201
Epoch 00015: val_loss did not improve from 1.45244
31/31 [==============================] - 3s 96ms/step - loss: 0.4150 - accuracy: 0.8201 - val_loss: 3.2344 - val_accuracy: 0.5513

Epoch 16/100
31/31 [==============================] - ETA: 0s - loss: 0.3838 - accuracy: 0.8216
Epoch 00016: val_loss improved from 1.45244 to 0.86896, saving model to model.h5
31/31 [==============================] - 4s 132ms/step - loss: 0.3838 - accuracy: 0.8216 - val_loss: 0.8690 - val_accuracy: 0.5030

Epoch 17/100
31/31 [==============================] - ETA: 0s - loss: 0.4117 - accuracy: 0.8327
Epoch 00017: val_loss did not improve from 0.86896
31/31 [==============================] - 3s 101ms/step - loss: 0.4117 - accuracy: 0.8327 - val_loss: 1.0979 - val_accuracy: 0.5513

Epoch 18/100
31/31 [==============================] - ETA: 0s - loss: 0.4132 - accuracy: 0.8276
Epoch 00018: val_loss improved from 0.86896 to 0.69180, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.4132 - accuracy: 0.8276 - val_loss: 0.6918 - val_accuracy: 0.5533

Epoch 19/100
31/31 [==============================] - ETA: 0s - loss: 0.3775 - accuracy: 0.8311
Epoch 00019: val_loss improved from 0.69180 to 0.66311, saving model to model.h5
31/31 [==============================] - 4s 140ms/step - loss: 0.3775 - accuracy: 0.8311 - val_loss: 0.6631 - val_accuracy: 0.5775

Epoch 20/100
31/31 [==============================] - ETA: 0s - loss: 0.3969 - accuracy: 0.8327
Epoch 00020: val_loss improved from 0.66311 to 0.64736, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.3969 - accuracy: 0.8327 - val_loss: 0.6474 - val_accuracy: 0.6016

Epoch 21/100
31/31 [==============================] - ETA: 0s - loss: 0.3692 - accuracy: 0.8493
Epoch 00021: val_loss improved from 0.64736 to 0.61495, saving model to model.h5
31/31 [==============================] - 4s 138ms/step - loss: 0.3692 - accuracy: 0.8493 - val_loss: 0.6150 - val_accuracy: 0.6761

Epoch 22/100
31/31 [==============================] - ETA: 0s - loss: 0.3691 - accuracy: 0.8453
Epoch 00022: val_loss did not improve from 0.61495
31/31 [==============================] - 4s 121ms/step - loss: 0.3691 - accuracy: 0.8453 - val_loss: 0.6224 - val_accuracy: 0.6338

Epoch 23/100
31/31 [==============================] - ETA: 0s - loss: 0.3648 - accuracy: 0.8448
Epoch 00023: val_loss improved from 0.61495 to 0.61105, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.3648 - accuracy: 0.8448 - val_loss: 0.6110 - val_accuracy: 0.6680

Epoch 24/100
31/31 [==============================] - ETA: 0s - loss: 0.3633 - accuracy: 0.8463
Epoch 00024: val_loss did not improve from 0.61105
31/31 [==============================] - 3s 96ms/step - loss: 0.3633 - accuracy: 0.8463 - val_loss: 0.6246 - val_accuracy: 0.6338

Epoch 25/100
31/31 [==============================] - ETA: 0s - loss: 0.3526 - accuracy: 0.8438
Epoch 00025: val_loss did not improve from 0.61105
31/31 [==============================] - 3s 107ms/step - loss: 0.3526 - accuracy: 0.8438 - val_loss: 0.6163 - val_accuracy: 0.6519

Epoch 26/100
31/31 [==============================] - ETA: 0s - loss: 0.3359 - accuracy: 0.8553
Epoch 00026: val_loss improved from 0.61105 to 0.55062, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.3359 - accuracy: 0.8553 - val_loss: 0.5506 - val_accuracy: 0.7264

Epoch 27/100
31/31 [==============================] - ETA: 0s - loss: 0.3360 - accuracy: 0.8604
Epoch 00027: val_loss improved from 0.55062 to 0.53647, saving model to model.h5
31/31 [==============================] - 5s 171ms/step - loss: 0.3360 - accuracy: 0.8604 - val_loss: 0.5365 - val_accuracy: 0.7123

Epoch 28/100
31/31 [==============================] - ETA: 0s - loss: 0.3257 - accuracy: 0.8664
Epoch 00028: val_loss improved from 0.53647 to 0.50145, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.3257 - accuracy: 0.8664 - val_loss: 0.5015 - val_accuracy: 0.7626

Epoch 29/100
31/31 [==============================] - ETA: 0s - loss: 0.3153 - accuracy: 0.8654
Epoch 00029: val_loss improved from 0.50145 to 0.48864, saving model to model.h5
31/31 [==============================] - 4s 133ms/step - loss: 0.3153 - accuracy: 0.8654 - val_loss: 0.4886 - val_accuracy: 0.7626

Epoch 30/100
31/31 [==============================] - ETA: 0s - loss: 0.3167 - accuracy: 0.8679
Epoch 00030: val_loss improved from 0.48864 to 0.47768, saving model to model.h5
31/31 [==============================] - 4s 145ms/step - loss: 0.3167 - accuracy: 0.8679 - val_loss: 0.4777 - val_accuracy: 0.7565

Epoch 31/100
31/31 [==============================] - ETA: 0s - loss: 0.3113 - accuracy: 0.8700
Epoch 00031: val_loss improved from 0.47768 to 0.43511, saving model to model.h5
31/31 [==============================] - 4s 133ms/step - loss: 0.3113 - accuracy: 0.8700 - val_loss: 0.4351 - val_accuracy: 0.7827

Epoch 32/100
31/31 [==============================] - ETA: 0s - loss: 0.3219 - accuracy: 0.8634
Epoch 00032: val_loss improved from 0.43511 to 0.41982, saving model to model.h5
31/31 [==============================] - 5s 176ms/step - loss: 0.3219 - accuracy: 0.8634 - val_loss: 0.4198 - val_accuracy: 0.7847

Epoch 33/100
31/31 [==============================] - ETA: 0s - loss: 0.3171 - accuracy: 0.8674
Epoch 00033: val_loss did not improve from 0.41982
31/31 [==============================] - 3s 96ms/step - loss: 0.3171 - accuracy: 0.8674 - val_loss: 0.5310 - val_accuracy: 0.7304

Epoch 34/100
31/31 [==============================] - ETA: 0s - loss: 0.3015 - accuracy: 0.8705
Epoch 00034: val_loss improved from 0.41982 to 0.41760, saving model to model.h5
31/31 [==============================] - 4s 132ms/step - loss: 0.3015 - accuracy: 0.8705 - val_loss: 0.4176 - val_accuracy: 0.7787

Epoch 35/100
31/31 [==============================] - ETA: 0s - loss: 0.3247 - accuracy: 0.8659
Epoch 00035: val_loss improved from 0.41760 to 0.41057, saving model to model.h5
31/31 [==============================] - 4s 142ms/step - loss: 0.3247 - accuracy: 0.8659 - val_loss: 0.4106 - val_accuracy: 0.8008

Epoch 36/100
31/31 [==============================] - ETA: 0s - loss: 0.3005 - accuracy: 0.8730
Epoch 00036: val_loss did not improve from 0.41057
31/31 [==============================] - 3s 96ms/step - loss: 0.3005 - accuracy: 0.8730 - val_loss: 0.4865 - val_accuracy: 0.7746

Epoch 37/100
31/31 [==============================] - ETA: 0s - loss: 0.3253 - accuracy: 0.8564
Epoch 00037: val_loss improved from 0.41057 to 0.31435, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.3253 - accuracy: 0.8564 - val_loss: 0.3143 - val_accuracy: 0.8491

Epoch 38/100
31/31 [==============================] - ETA: 0s - loss: 0.3135 - accuracy: 0.8684
Epoch 00038: val_loss improved from 0.31435 to 0.29319, saving model to model.h5
31/31 [==============================] - 5s 171ms/step - loss: 0.3135 - accuracy: 0.8684 - val_loss: 0.2932 - val_accuracy: 0.8692

Epoch 39/100
31/31 [==============================] - ETA: 0s - loss: 0.3023 - accuracy: 0.8800
Epoch 00039: val_loss did not improve from 0.29319
31/31 [==============================] - 3s 95ms/step - loss: 0.3023 - accuracy: 0.8800 - val_loss: 0.3662 - val_accuracy: 0.8169

Epoch 40/100
31/31 [==============================] - ETA: 0s - loss: 0.3011 - accuracy: 0.8770
Epoch 00040: val_loss improved from 0.29319 to 0.28929, saving model to model.h5
31/31 [==============================] - 4s 141ms/step - loss: 0.3011 - accuracy: 0.8770 - val_loss: 0.2893 - val_accuracy: 0.8712

Epoch 41/100
31/31 [==============================] - ETA: 0s - loss: 0.2955 - accuracy: 0.8821
Epoch 00041: val_loss improved from 0.28929 to 0.20949, saving model to model.h5
31/31 [==============================] - 5s 172ms/step - loss: 0.2955 - accuracy: 0.8821 - val_loss: 0.2095 - val_accuracy: 0.9195

Epoch 42/100
31/31 [==============================] - ETA: 0s - loss: 0.2872 - accuracy: 0.8821
Epoch 00042: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 95ms/step - loss: 0.2872 - accuracy: 0.8821 - val_loss: 0.3225 - val_accuracy: 0.8410

Epoch 43/100
31/31 [==============================] - ETA: 0s - loss: 0.2841 - accuracy: 0.8816
Epoch 00043: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 102ms/step - loss: 0.2841 - accuracy: 0.8816 - val_loss: 0.2451 - val_accuracy: 0.8893

Epoch 44/100
31/31 [==============================] - ETA: 0s - loss: 0.2768 - accuracy: 0.8831
Epoch 00044: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 98ms/step - loss: 0.2768 - accuracy: 0.8831 - val_loss: 0.2400 - val_accuracy: 0.8893

Epoch 45/100
31/31 [==============================] - ETA: 0s - loss: 0.2872 - accuracy: 0.8730
Epoch 00045: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 95ms/step - loss: 0.2872 - accuracy: 0.8730 - val_loss: 0.2129 - val_accuracy: 0.9074

Epoch 46/100
31/31 [==============================] - ETA: 0s - loss: 0.2698 - accuracy: 0.8967
Epoch 00046: ReduceLROnPlateau reducing learning rate to 0.0010289999307133257.

Epoch 00046: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 104ms/step - loss: 0.2698 - accuracy: 0.8967 - val_loss: 0.2646 - val_accuracy: 0.8773

Epoch 47/100
31/31 [==============================] - ETA: 0s - loss: 0.2682 - accuracy: 0.8775
Epoch 00047: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 95ms/step - loss: 0.2682 - accuracy: 0.8775 - val_loss: 0.4491 - val_accuracy: 0.8129

Epoch 48/100
31/31 [==============================] - ETA: 0s - loss: 0.2574 - accuracy: 0.8851
Epoch 00048: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 96ms/step - loss: 0.2574 - accuracy: 0.8851 - val_loss: 0.2930 - val_accuracy: 0.8732

Epoch 49/100
31/31 [==============================] - ETA: 0s - loss: 0.2746 - accuracy: 0.8876
Epoch 00049: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 101ms/step - loss: 0.2746 - accuracy: 0.8876 - val_loss: 0.2216 - val_accuracy: 0.9095

Epoch 50/100
31/31 [==============================] - ETA: 0s - loss: 0.2846 - accuracy: 0.8836
Epoch 00050: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 96ms/step - loss: 0.2846 - accuracy: 0.8836 - val_loss: 0.2179 - val_accuracy: 0.9074

Epoch 51/100
31/31 [==============================] - ETA: 0s - loss: 0.2552 - accuracy: 0.8921
Epoch 00051: ReduceLROnPlateau reducing learning rate to 0.0007202999433502554.

Epoch 00051: val_loss did not improve from 0.20949
31/31 [==============================] - 4s 114ms/step - loss: 0.2552 - accuracy: 0.8921 - val_loss: 0.5933 - val_accuracy: 0.7626

Epoch 52/100
31/31 [==============================] - ETA: 0s - loss: 0.2626 - accuracy: 0.8816
Epoch 00052: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 96ms/step - loss: 0.2626 - accuracy: 0.8816 - val_loss: 0.2280 - val_accuracy: 0.8934

Epoch 53/100
31/31 [==============================] - ETA: 0s - loss: 0.2684 - accuracy: 0.8906
Epoch 00053: val_loss did not improve from 0.20949
31/31 [==============================] - 3s 105ms/step - loss: 0.2684 - accuracy: 0.8906 - val_loss: 0.2276 - val_accuracy: 0.9074

Epoch 54/100
31/31 [==============================] - ETA: 0s - loss: 0.2703 - accuracy: 0.8805
Epoch 00054: val_loss improved from 0.20949 to 0.18132, saving model to model.h5
31/31 [==============================] - 4s 131ms/step - loss: 0.2703 - accuracy: 0.8805 - val_loss: 0.1813 - val_accuracy: 0.9235

Epoch 55/100
31/31 [==============================] - ETA: 0s - loss: 0.2603 - accuracy: 0.8926
Epoch 00055: val_loss did not improve from 0.18132
31/31 [==============================] - 3s 95ms/step - loss: 0.2603 - accuracy: 0.8926 - val_loss: 0.1863 - val_accuracy: 0.9256

Epoch 56/100
31/31 [==============================] - ETA: 0s - loss: 0.2702 - accuracy: 0.8891
Epoch 00056: val_loss did not improve from 0.18132
31/31 [==============================] - 3s 106ms/step - loss: 0.2702 - accuracy: 0.8891 - val_loss: 0.4952 - val_accuracy: 0.7928

Epoch 57/100
31/31 [==============================] - ETA: 0s - loss: 0.2564 - accuracy: 0.8926
Epoch 00057: val_loss improved from 0.18132 to 0.17801, saving model to model.h5
31/31 [==============================] - 4s 135ms/step - loss: 0.2564 - accuracy: 0.8926 - val_loss: 0.1780 - val_accuracy: 0.9276

Epoch 58/100
31/31 [==============================] - ETA: 0s - loss: 0.2778 - accuracy: 0.8856
Epoch 00058: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 95ms/step - loss: 0.2778 - accuracy: 0.8856 - val_loss: 0.2364 - val_accuracy: 0.8833

Epoch 59/100
31/31 [==============================] - ETA: 0s - loss: 0.2559 - accuracy: 0.8891
Epoch 00059: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 107ms/step - loss: 0.2559 - accuracy: 0.8891 - val_loss: 0.2141 - val_accuracy: 0.8974

Epoch 60/100
31/31 [==============================] - ETA: 0s - loss: 0.2584 - accuracy: 0.8891
Epoch 00060: val_loss did not improve from 0.17801
31/31 [==============================] - 4s 116ms/step - loss: 0.2584 - accuracy: 0.8891 - val_loss: 0.3432 - val_accuracy: 0.8290

Epoch 61/100
31/31 [==============================] - ETA: 0s - loss: 0.2622 - accuracy: 0.8916
Epoch 00061: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 95ms/step - loss: 0.2622 - accuracy: 0.8916 - val_loss: 0.2799 - val_accuracy: 0.8692

Epoch 62/100
31/31 [==============================] - ETA: 0s - loss: 0.2483 - accuracy: 0.8947
Epoch 00062: ReduceLROnPlateau reducing learning rate to 0.0005042099684942513.

Epoch 00062: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 105ms/step - loss: 0.2483 - accuracy: 0.8947 - val_loss: 0.1943 - val_accuracy: 0.9095

Epoch 63/100
31/31 [==============================] - ETA: 0s - loss: 0.2592 - accuracy: 0.8891
Epoch 00063: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 96ms/step - loss: 0.2592 - accuracy: 0.8891 - val_loss: 0.1807 - val_accuracy: 0.9215

Epoch 64/100
31/31 [==============================] - ETA: 0s - loss: 0.2521 - accuracy: 0.8906
Epoch 00064: val_loss did not improve from 0.17801
31/31 [==============================] - 3s 95ms/step - loss: 0.2521 - accuracy: 0.8906 - val_loss: 0.1829 - val_accuracy: 0.9235

Epoch 65/100
31/31 [==============================] - ETA: 0s - loss: 0.2507 - accuracy: 0.8911
Epoch 00065: val_loss improved from 0.17801 to 0.17014, saving model to model.h5
31/31 [==============================] - 4s 137ms/step - loss: 0.2507 - accuracy: 0.8911 - val_loss: 0.1701 - val_accuracy: 0.9276

Epoch 66/100
31/31 [==============================] - ETA: 0s - loss: 0.2351 - accuracy: 0.9022
Epoch 00066: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 96ms/step - loss: 0.2351 - accuracy: 0.9022 - val_loss: 0.1901 - val_accuracy: 0.9155

Epoch 67/100
31/31 [==============================] - ETA: 0s - loss: 0.2603 - accuracy: 0.8896
Epoch 00067: ReduceLROnPlateau reducing learning rate to 0.0003529469657223671.

Epoch 00067: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 102ms/step - loss: 0.2603 - accuracy: 0.8896 - val_loss: 0.2167 - val_accuracy: 0.8853

Epoch 68/100
31/31 [==============================] - ETA: 0s - loss: 0.2464 - accuracy: 0.8936
Epoch 00068: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 98ms/step - loss: 0.2464 - accuracy: 0.8936 - val_loss: 0.1829 - val_accuracy: 0.9175

Epoch 69/100
31/31 [==============================] - ETA: 0s - loss: 0.2380 - accuracy: 0.8942
Epoch 00069: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 112ms/step - loss: 0.2380 - accuracy: 0.8942 - val_loss: 0.1748 - val_accuracy: 0.9256

Epoch 70/100
31/31 [==============================] - ETA: 0s - loss: 0.2429 - accuracy: 0.8997
Epoch 00070: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 99ms/step - loss: 0.2429 - accuracy: 0.8997 - val_loss: 0.1778 - val_accuracy: 0.9215

Epoch 71/100
31/31 [==============================] - ETA: 0s - loss: 0.2566 - accuracy: 0.8901
Epoch 00071: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 96ms/step - loss: 0.2566 - accuracy: 0.8901 - val_loss: 0.1713 - val_accuracy: 0.9256

Epoch 72/100
31/31 [==============================] - ETA: 0s - loss: 0.2492 - accuracy: 0.8967
Epoch 00072: ReduceLROnPlateau reducing learning rate to 0.0002470628678565845.

Epoch 00072: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 109ms/step - loss: 0.2492 - accuracy: 0.8967 - val_loss: 0.2305 - val_accuracy: 0.8793

Epoch 73/100
31/31 [==============================] - ETA: 0s - loss: 0.2436 - accuracy: 0.9007
Epoch 00073: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 96ms/step - loss: 0.2436 - accuracy: 0.9007 - val_loss: 0.1905 - val_accuracy: 0.9155

Epoch 74/100
31/31 [==============================] - ETA: 0s - loss: 0.2570 - accuracy: 0.8942
Epoch 00074: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 96ms/step - loss: 0.2570 - accuracy: 0.8942 - val_loss: 0.1711 - val_accuracy: 0.9256

Epoch 75/100
31/31 [==============================] - ETA: 0s - loss: 0.2452 - accuracy: 0.8931
Epoch 00075: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 106ms/step - loss: 0.2452 - accuracy: 0.8931 - val_loss: 0.1974 - val_accuracy: 0.9054

Epoch 76/100
31/31 [==============================] - ETA: 0s - loss: 0.2451 - accuracy: 0.8896
Epoch 00076: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 95ms/step - loss: 0.2451 - accuracy: 0.8896 - val_loss: 0.2238 - val_accuracy: 0.8853

Epoch 77/100
31/31 [==============================] - ETA: 0s - loss: 0.2416 - accuracy: 0.8952
Epoch 00077: ReduceLROnPlateau reducing learning rate to 0.00017294401768594978.

Epoch 00077: val_loss did not improve from 0.17014
31/31 [==============================] - 3s 96ms/step - loss: 0.2416 - accuracy: 0.8952 - val_loss: 0.1707 - val_accuracy: 0.9276

Epoch 78/100
31/31 [==============================] - ETA: 0s - loss: 0.2478 - accuracy: 0.8936
Epoch 00078: val_loss improved from 0.17014 to 0.16120, saving model to model.h5
31/31 [==============================] - 4s 139ms/step - loss: 0.2478 - accuracy: 0.8936 - val_loss: 0.1612 - val_accuracy: 0.9296

Epoch 79/100
31/31 [==============================] - ETA: 0s - loss: 0.2451 - accuracy: 0.8962
Epoch 00079: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 96ms/step - loss: 0.2451 - accuracy: 0.8962 - val_loss: 0.1667 - val_accuracy: 0.9316

Epoch 80/100
31/31 [==============================] - ETA: 0s - loss: 0.2266 - accuracy: 0.8982
Epoch 00080: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 96ms/step - loss: 0.2266 - accuracy: 0.8982 - val_loss: 0.1659 - val_accuracy: 0.9276

Epoch 81/100
31/31 [==============================] - ETA: 0s - loss: 0.2445 - accuracy: 0.9012
Epoch 00081: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 96ms/step - loss: 0.2445 - accuracy: 0.9012 - val_loss: 0.1697 - val_accuracy: 0.9235

Epoch 82/100
31/31 [==============================] - ETA: 0s - loss: 0.2272 - accuracy: 0.9017
Epoch 00082: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 103ms/step - loss: 0.2272 - accuracy: 0.9017 - val_loss: 0.1703 - val_accuracy: 0.9235

Epoch 83/100
31/31 [==============================] - ETA: 0s - loss: 0.2419 - accuracy: 0.8992
Epoch 00083: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 113ms/step - loss: 0.2419 - accuracy: 0.8992 - val_loss: 0.1717 - val_accuracy: 0.9296

Epoch 84/100
31/31 [==============================] - ETA: 0s - loss: 0.2437 - accuracy: 0.8936
Epoch 00084: ReduceLROnPlateau reducing learning rate to 0.00012106080830562859.

Epoch 00084: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 96ms/step - loss: 0.2437 - accuracy: 0.8936 - val_loss: 0.1936 - val_accuracy: 0.9054

Epoch 85/100
31/31 [==============================] - ETA: 0s - loss: 0.2483 - accuracy: 0.8936
Epoch 00085: val_loss did not improve from 0.16120
31/31 [==============================] - 3s 104ms/step - loss: 0.2483 - accuracy: 0.8936 - val_loss: 0.1758 - val_accuracy: 0.9155

Epoch 86/100
31/31 [==============================] - ETA: 0s - loss: 0.2271 - accuracy: 0.9012
Epoch 00086: val_loss improved from 0.16120 to 0.16092, saving model to model.h5
31/31 [==============================] - 4s 133ms/step - loss: 0.2271 - accuracy: 0.9012 - val_loss: 0.1609 - val_accuracy: 0.9256

Epoch 87/100
31/31 [==============================] - ETA: 0s - loss: 0.2330 - accuracy: 0.9052
Epoch 00087: val_loss did not improve from 0.16092
31/31 [==============================] - 3s 95ms/step - loss: 0.2330 - accuracy: 0.9052 - val_loss: 0.1614 - val_accuracy: 0.9276

Epoch 88/100
31/31 [==============================] - ETA: 0s - loss: 0.2531 - accuracy: 0.8891
Epoch 00088: val_loss improved from 0.16092 to 0.16062, saving model to model.h5
31/31 [==============================] - 4s 142ms/step - loss: 0.2531 - accuracy: 0.8891 - val_loss: 0.1606 - val_accuracy: 0.9296

Epoch 89/100
31/31 [==============================] - ETA: 0s - loss: 0.2369 - accuracy: 0.8982
Epoch 00089: ReduceLROnPlateau reducing learning rate to 0.0001.

Epoch 00089: val_loss improved from 0.16062 to 0.15927, saving model to model.h5
31/31 [==============================] - 5s 168ms/step - loss: 0.2369 - accuracy: 0.8982 - val_loss: 0.1593 - val_accuracy: 0.9296

Epoch 90/100
31/31 [==============================] - ETA: 0s - loss: 0.2335 - accuracy: 0.9042
Epoch 00090: val_loss improved from 0.15927 to 0.15853, saving model to model.h5
31/31 [==============================] - 5s 145ms/step - loss: 0.2335 - accuracy: 0.9042 - val_loss: 0.1585 - val_accuracy: 0.9276

Epoch 91/100
31/31 [==============================] - ETA: 0s - loss: 0.2274 - accuracy: 0.9083
Epoch 00091: val_loss did not improve from 0.15853
31/31 [==============================] - 4s 117ms/step - loss: 0.2274 - accuracy: 0.9083 - val_loss: 0.1655 - val_accuracy: 0.9235

Epoch 92/100
31/31 [==============================] - ETA: 0s - loss: 0.2346 - accuracy: 0.9052
Epoch 00092: val_loss improved from 0.15853 to 0.15741, saving model to model.h5
31/31 [==============================] - 4s 135ms/step - loss: 0.2346 - accuracy: 0.9052 - val_loss: 0.1574 - val_accuracy: 0.9256

Epoch 93/100
31/31 [==============================] - ETA: 0s - loss: 0.2390 - accuracy: 0.9027
Epoch 00093: val_loss improved from 0.15741 to 0.15642, saving model to model.h5
31/31 [==============================] - 5s 167ms/step - loss: 0.2390 - accuracy: 0.9027 - val_loss: 0.1564 - val_accuracy: 0.9276

Epoch 94/100
31/31 [==============================] - ETA: 0s - loss: 0.2491 - accuracy: 0.8962
Epoch 00094: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 96ms/step - loss: 0.2491 - accuracy: 0.8962 - val_loss: 0.1660 - val_accuracy: 0.9256

Epoch 95/100
31/31 [==============================] - ETA: 0s - loss: 0.2367 - accuracy: 0.9002
Epoch 00095: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 98ms/step - loss: 0.2367 - accuracy: 0.9002 - val_loss: 0.1700 - val_accuracy: 0.9256

Epoch 96/100
31/31 [==============================] - ETA: 0s - loss: 0.2502 - accuracy: 0.8926
Epoch 00096: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 105ms/step - loss: 0.2502 - accuracy: 0.8926 - val_loss: 0.1703 - val_accuracy: 0.9195

Epoch 97/100
31/31 [==============================] - ETA: 0s - loss: 0.2262 - accuracy: 0.9042
Epoch 00097: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 95ms/step - loss: 0.2262 - accuracy: 0.9042 - val_loss: 0.1842 - val_accuracy: 0.9074

Epoch 98/100
31/31 [==============================] - ETA: 0s - loss: 0.2206 - accuracy: 0.9057
Epoch 00098: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 95ms/step - loss: 0.2206 - accuracy: 0.9057 - val_loss: 0.1932 - val_accuracy: 0.9054

Epoch 99/100
31/31 [==============================] - ETA: 0s - loss: 0.2227 - accuracy: 0.9042
Epoch 00099: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 105ms/step - loss: 0.2227 - accuracy: 0.9042 - val_loss: 0.1845 - val_accuracy: 0.9155

Epoch 100/100
31/31 [==============================] - ETA: 0s - loss: 0.2426 - accuracy: 0.8977
Epoch 00100: val_loss did not improve from 0.15642
31/31 [==============================] - 4s 115ms/step - loss: 0.2426 - accuracy: 0.8977 - val_loss: 0.1877 - val_accuracy: 0.9115

```
# 불필요한 반복 (코딩 수정 중 오류)
# 한번 더 훈련시키므로 예측값이 처음부터 좋을 수 밖에 없다.
# Fits the model on batches with real-time data augmentation
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))
```

Epoch 1/100
31/31 [==============================] - ETA: 0s - loss: 0.2345 - accuracy: 0.9042
Epoch 00001: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 97ms/step - loss: 0.2345 - accuracy: 0.9042 - val_loss: 0.1807 - val_accuracy: 0.9195

Epoch 2/100
31/31 [==============================] - ETA: 0s - loss: 0.2437 - accuracy: 0.8906
Epoch 00002: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 106ms/step - loss: 0.2437 - accuracy: 0.8906 - val_loss: 0.1679 - val_accuracy: 0.9195

Epoch 3/100
31/31 [==============================] - ETA: 0s - loss: 0.2300 - accuracy: 0.9037
Epoch 00003: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 96ms/step - loss: 0.2300 - accuracy: 0.9037 - val_loss: 0.1589 - val_accuracy: 0.9316

Epoch 4/100
31/31 [==============================] - ETA: 0s - loss: 0.2393 - accuracy: 0.8967
Epoch 00004: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 96ms/step - loss: 0.2393 - accuracy: 0.8967 - val_loss: 0.1585 - val_accuracy: 0.9276

Epoch 5/100
31/31 [==============================] - ETA: 0s - loss: 0.2144 - accuracy: 0.9113
Epoch 00005: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 96ms/step - loss: 0.2144 - accuracy: 0.9113 - val_loss: 0.1685 - val_accuracy: 0.9235

Epoch 6/100
31/31 [==============================] - ETA: 0s - loss: 0.2415 - accuracy: 0.8947
Epoch 00006: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 105ms/step - loss: 0.2415 - accuracy: 0.8947 - val_loss: 0.1683 - val_accuracy: 0.9235

Epoch 7/100
31/31 [==============================] - ETA: 0s - loss: 0.2365 - accuracy: 0.9012
Epoch 00007: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 95ms/step - loss: 0.2365 - accuracy: 0.9012 - val_loss: 0.1572 - val_accuracy: 0.9276

Epoch 8/100
31/31 [==============================] - ETA: 0s - loss: 0.2295 - accuracy: 0.9022
Epoch 00008: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 95ms/step - loss: 0.2295 - accuracy: 0.9022 - val_loss: 0.1577 - val_accuracy: 0.9215

Epoch 9/100
31/31 [==============================] - ETA: 0s - loss: 0.2356 - accuracy: 0.9108
Epoch 00009: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 106ms/step - loss: 0.2356 - accuracy: 0.9108 - val_loss: 0.1571 - val_accuracy: 0.9235

Epoch 10/100
31/31 [==============================] - ETA: 0s - loss: 0.2145 - accuracy: 0.9047
Epoch 00010: val_loss did not improve from 0.15642
31/31 [==============================] - 4s 115ms/step - loss: 0.2145 - accuracy: 0.9047 - val_loss: 0.1572 - val_accuracy: 0.9215

Epoch 11/100
31/31 [==============================] - ETA: 0s - loss: 0.2393 - accuracy: 0.8972
Epoch 00011: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 95ms/step - loss: 0.2393 - accuracy: 0.8972 - val_loss: 0.1639 - val_accuracy: 0.9256

Epoch 12/100
31/31 [==============================] - ETA: 0s - loss: 0.2309 - accuracy: 0.9022
Epoch 00012: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 100ms/step - loss: 0.2309 - accuracy: 0.9022 - val_loss: 0.1671 - val_accuracy: 0.9155

Epoch 13/100
31/31 [==============================] - ETA: 0s - loss: 0.2392 - accuracy: 0.8977
Epoch 00013: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 96ms/step - loss: 0.2392 - accuracy: 0.8977 - val_loss: 0.1669 - val_accuracy: 0.9215

Epoch 14/100
31/31 [==============================] - ETA: 0s - loss: 0.2257 - accuracy: 0.9098
Epoch 00014: val_loss did not improve from 0.15642
31/31 [==============================] - 3s 97ms/step - loss: 0.2257 - accuracy: 0.9098 - val_loss: 0.1605 - val_accuracy: 0.9235

Epoch 15/100
31/31 [==============================] - ETA: 0s - loss: 0.2271 - accuracy: 0.9073
Epoch 00015: val_loss improved from 0.15642 to 0.15639, saving model to model.h5
31/31 [==============================] - 4s 134ms/step - loss: 0.2271 - accuracy: 0.9073 - val_loss: 0.1564 - val_accuracy: 0.9235

Epoch 16/100
31/31 [==============================] - ETA: 0s - loss: 0.2255 - accuracy: 0.9007
Epoch 00016: val_loss did not improve from 0.15639
31/31 [==============================] - 3s 100ms/step - loss: 0.2255 - accuracy: 0.9007 - val_loss: 0.1581 - val_accuracy: 0.9235

Epoch 17/100
31/31 [==============================] - ETA: 0s - loss: 0.2311 - accuracy: 0.9052
Epoch 00017: val_loss did not improve from 0.15639
31/31 [==============================] - 3s 98ms/step - loss: 0.2311 - accuracy: 0.9052 - val_loss: 0.1574 - val_accuracy: 0.9195

Epoch 18/100
31/31 [==============================] - ETA: 0s - loss: 0.2310 - accuracy: 0.9022
Epoch 00018: val_loss did not improve from 0.15639
31/31 [==============================] - 3s 97ms/step - loss: 0.2310 - accuracy: 0.9022 - val_loss: 0.1571 - val_accuracy: 0.9195

Epoch 19/100
31/31 [==============================] - ETA: 0s - loss: 0.2249 - accuracy: 0.9148
Epoch 00019: val_loss did not improve from 0.15639
31/31 [==============================] - 3s 108ms/step - loss: 0.2249 - accuracy: 0.9148 - val_loss: 0.1589 - val_accuracy: 0.9235

Epoch 20/100
31/31 [==============================] - ETA: 0s - loss: 0.2282 - accuracy: 0.9037
Epoch 00020: val_loss did not improve from 0.15639
31/31 [==============================] - 3s 112ms/step - loss: 0.2282 - accuracy: 0.9037 - val_loss: 0.1572 - val_accuracy: 0.9256

Epoch 21/100
31/31 [==============================] - ETA: 0s - loss: 0.2405 - accuracy: 0.9088
Epoch 00021: val_loss improved from 0.15639 to 0.15581, saving model to model.h5
31/31 [==============================] - 4s 132ms/step - loss: 0.2405 - accuracy: 0.9088 - val_loss: 0.1558 - val_accuracy: 0.9235

Epoch 22/100
31/31 [==============================] - ETA: 0s - loss: 0.2318 - accuracy: 0.9037
Epoch 00022: val_loss did not improve from 0.15581
31/31 [==============================] - 3s 104ms/step - loss: 0.2318 - accuracy: 0.9037 - val_loss: 0.1561 - val_accuracy: 0.9215

Epoch 23/100
31/31 [==============================] - ETA: 0s - loss: 0.2319 - accuracy: 0.8972
Epoch 00023: val_loss did not improve from 0.15581
31/31 [==============================] - 3s 97ms/step - loss: 0.2319 - accuracy: 0.8972 - val_loss: 0.1560 - val_accuracy: 0.9235

Epoch 24/100
31/31 [==============================] - ETA: 0s - loss: 0.2311 - accuracy: 0.9017
Epoch 00024: val_loss did not improve from 0.15581
31/31 [==============================] - 3s 95ms/step - loss: 0.2311 - accuracy: 0.9017 - val_loss: 0.1589 - val_accuracy: 0.9316
Epoch 25/100
31/31 [==============================] - ETA: 0s - loss: 0.2401 - accuracy: 0.9022
Epoch 00025: val_loss improved from 0.15581 to 0.15480, saving model to model.h5
31/31 [==============================] - 4s 141ms/step - loss: 0.2401 - accuracy: 0.9022 - val_loss: 0.1548 - val_accuracy: 0.9296

Epoch 26/100
31/31 [==============================] - ETA: 0s - loss: 0.2405 - accuracy: 0.8957
Epoch 00026: val_loss improved from 0.15480 to 0.15372, saving model to model.h5
31/31 [==============================] - 5s 166ms/step - loss: 0.2405 - accuracy: 0.8957 - val_loss: 0.1537 - val_accuracy: 0.9296

Epoch 27/100
31/31 [==============================] - ETA: 0s - loss: 0.2229 - accuracy: 0.9052
Epoch 00027: val_loss improved from 0.15372 to 0.15358, saving model to model.h5
31/31 [==============================] - 4s 139ms/step - loss: 0.2229 - accuracy: 0.9052 - val_loss: 0.1536 - val_accuracy: 0.9336

Epoch 28/100
31/31 [==============================] - ETA: 0s - loss: 0.2307 - accuracy: 0.9057
Epoch 00028: val_loss did not improve from 0.15358
31/31 [==============================] - 3s 96ms/step - loss: 0.2307 - accuracy: 0.9057 - val_loss: 0.1554 - val_accuracy: 0.9296

Epoch 29/100
31/31 [==============================] - ETA: 0s - loss: 0.2256 - accuracy: 0.9088
Epoch 00029: val_loss improved from 0.15358 to 0.15259, saving model to model.h5
31/31 [==============================] - 4s 133ms/step - loss: 0.2256 - accuracy: 0.9088 - val_loss: 0.1526 - val_accuracy: 0.9336

Epoch 30/100
31/31 [==============================] - ETA: 0s - loss: 0.2215 - accuracy: 0.9057
Epoch 00030: val_loss improved from 0.15259 to 0.15197, saving model to model.h5
31/31 [==============================] - 5s 171ms/step - loss: 0.2215 - accuracy: 0.9057 - val_loss: 0.1520 - val_accuracy: 0.9296

Epoch 31/100
31/31 [==============================] - ETA: 0s - loss: 0.2102 - accuracy: 0.9103
Epoch 00031: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2102 - accuracy: 0.9103 - val_loss: 0.1616 - val_accuracy: 0.9276

Epoch 32/100
31/31 [==============================] - ETA: 0s - loss: 0.2129 - accuracy: 0.9073
Epoch 00032: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2129 - accuracy: 0.9073 - val_loss: 0.1531 - val_accuracy: 0.9316

Epoch 33/100
31/31 [==============================] - ETA: 0s - loss: 0.2329 - accuracy: 0.8997
Epoch 00033: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 104ms/step - loss: 0.2329 - accuracy: 0.8997 - val_loss: 0.1530 - val_accuracy: 0.9336

Epoch 34/100
31/31 [==============================] - ETA: 0s - loss: 0.2320 - accuracy: 0.9032
Epoch 00034: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2320 - accuracy: 0.9032 - val_loss: 0.1563 - val_accuracy: 0.9316

Epoch 35/100
31/31 [==============================] - ETA: 0s - loss: 0.2294 - accuracy: 0.8987
Epoch 00035: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2294 - accuracy: 0.8987 - val_loss: 0.1544 - val_accuracy: 0.9276

Epoch 36/100
31/31 [==============================] - ETA: 0s - loss: 0.2336 - accuracy: 0.9032
Epoch 00036: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 98ms/step - loss: 0.2336 - accuracy: 0.9032 - val_loss: 0.1546 - val_accuracy: 0.9296

Epoch 37/100
31/31 [==============================] - ETA: 0s - loss: 0.2271 - accuracy: 0.9062
Epoch 00037: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 99ms/step - loss: 0.2271 - accuracy: 0.9062 - val_loss: 0.1563 - val_accuracy: 0.9316

Epoch 38/100
31/31 [==============================] - ETA: 0s - loss: 0.2136 - accuracy: 0.9093
Epoch 00038: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2136 - accuracy: 0.9093 - val_loss: 0.1586 - val_accuracy: 0.9316

Epoch 39/100
31/31 [==============================] - ETA: 0s - loss: 0.2094 - accuracy: 0.9128
Epoch 00039: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2094 - accuracy: 0.9128 - val_loss: 0.1550 - val_accuracy: 0.9276

Epoch 40/100
31/31 [==============================] - ETA: 0s - loss: 0.2242 - accuracy: 0.9007
Epoch 00040: val_loss did not improve from 0.15197
31/31 [==============================] - 4s 126ms/step - loss: 0.2242 - accuracy: 0.9007 - val_loss: 0.1621 - val_accuracy: 0.9276

Epoch 41/100
31/31 [==============================] - ETA: 0s - loss: 0.2394 - accuracy: 0.8962
Epoch 00041: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2394 - accuracy: 0.8962 - val_loss: 0.1639 - val_accuracy: 0.9276

Epoch 42/100
31/31 [==============================] - ETA: 0s - loss: 0.2133 - accuracy: 0.9118
Epoch 00042: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2133 - accuracy: 0.9118 - val_loss: 0.1573 - val_accuracy: 0.9316

Epoch 43/100
31/31 [==============================] - ETA: 0s - loss: 0.2223 - accuracy: 0.9078
Epoch 00043: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 105ms/step - loss: 0.2223 - accuracy: 0.9078 - val_loss: 0.1543 - val_accuracy: 0.9316

Epoch 44/100
31/31 [==============================] - ETA: 0s - loss: 0.2113 - accuracy: 0.9088
Epoch 00044: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2113 - accuracy: 0.9088 - val_loss: 0.1575 - val_accuracy: 0.9256

Epoch 45/100
31/31 [==============================] - ETA: 0s - loss: 0.2317 - accuracy: 0.9022
Epoch 00045: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2317 - accuracy: 0.9022 - val_loss: 0.1606 - val_accuracy: 0.9296

Epoch 46/100
31/31 [==============================] - ETA: 0s - loss: 0.2205 - accuracy: 0.9052
Epoch 00046: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2205 - accuracy: 0.9052 - val_loss: 0.1671 - val_accuracy: 0.9215

Epoch 47/100
31/31 [==============================] - ETA: 0s - loss: 0.2233 - accuracy: 0.9098
Epoch 00047: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 105ms/step - loss: 0.2233 - accuracy: 0.9098 - val_loss: 0.1536 - val_accuracy: 0.9296

Epoch 48/100
31/31 [==============================] - ETA: 0s - loss: 0.2276 - accuracy: 0.9052
Epoch 00048: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2276 - accuracy: 0.9052 - val_loss: 0.1719 - val_accuracy: 0.9235

Epoch 49/100
31/31 [==============================] - ETA: 0s - loss: 0.2267 - accuracy: 0.9078
Epoch 00049: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2267 - accuracy: 0.9078 - val_loss: 0.1716 - val_accuracy: 0.9215

Epoch 50/100
31/31 [==============================] - ETA: 0s - loss: 0.2285 - accuracy: 0.8977
Epoch 00050: val_loss did not improve from 0.15197
31/31 [==============================] - 4s 124ms/step - loss: 0.2285 - accuracy: 0.8977 - val_loss: 0.1591 - val_accuracy: 0.9276

Epoch 51/100
31/31 [==============================] - ETA: 0s - loss: 0.2410 - accuracy: 0.9022
Epoch 00051: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2410 - accuracy: 0.9022 - val_loss: 0.1552 - val_accuracy: 0.9336

Epoch 52/100
31/31 [==============================] - ETA: 0s - loss: 0.2462 - accuracy: 0.8926
Epoch 00052: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2462 - accuracy: 0.8926 - val_loss: 0.1642 - val_accuracy: 0.9276

Epoch 53/100
31/31 [==============================] - ETA: 0s - loss: 0.2353 - accuracy: 0.8982
Epoch 00053: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 99ms/step - loss: 0.2353 - accuracy: 0.8982 - val_loss: 0.1723 - val_accuracy: 0.9195

Epoch 54/100
31/31 [==============================] - ETA: 0s - loss: 0.2230 - accuracy: 0.9047
Epoch 00054: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 96ms/step - loss: 0.2230 - accuracy: 0.9047 - val_loss: 0.1609 - val_accuracy: 0.9235
Epoch 55/100
31/31 [==============================] - ETA: 0s - loss: 0.2259 - accuracy: 0.9068
Epoch 00055: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2259 - accuracy: 0.9068 - val_loss: 0.1563 - val_accuracy: 0.9276

Epoch 56/100
31/31 [==============================] - ETA: 0s - loss: 0.2301 - accuracy: 0.9032
Epoch 00056: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2301 - accuracy: 0.9032 - val_loss: 0.1565 - val_accuracy: 0.9276

Epoch 57/100
31/31 [==============================] - ETA: 0s - loss: 0.2092 - accuracy: 0.9123
Epoch 00057: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 105ms/step - loss: 0.2092 - accuracy: 0.9123 - val_loss: 0.1602 - val_accuracy: 0.9276

Epoch 58/100
31/31 [==============================] - ETA: 0s - loss: 0.2263 - accuracy: 0.9012
Epoch 00058: val_loss did not improve from 0.15197
31/31 [==============================] - 3s 95ms/step - loss: 0.2263 - accuracy: 0.9012 - val_loss: 0.1545 - val_accuracy: 0.9296

Epoch 59/100
31/31 [==============================] - ETA: 0s - loss: 0.2254 - accuracy: 0.9052
Epoch 00059: val_loss improved from 0.15197 to 0.15149, saving model to model.h5
31/31 [==============================] - 4s 132ms/step - loss: 0.2254 - accuracy: 0.9052 - val_loss: 0.1515 - val_accuracy: 0.9336

Epoch 60/100
31/31 [==============================] - ETA: 0s - loss: 0.2214 - accuracy: 0.9078
Epoch 00060: val_loss did not improve from 0.15149
31/31 [==============================] - 4s 124ms/step - loss: 0.2214 - accuracy: 0.9078 - val_loss: 0.1576 - val_accuracy: 0.9296

Epoch 61/100
31/31 [==============================] - ETA: 0s - loss: 0.2101 - accuracy: 0.9083
Epoch 00061: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 98ms/step - loss: 0.2101 - accuracy: 0.9083 - val_loss: 0.1650 - val_accuracy: 0.9235

Epoch 62/100
31/31 [==============================] - ETA: 0s - loss: 0.2205 - accuracy: 0.9062
Epoch 00062: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 96ms/step - loss: 0.2205 - accuracy: 0.9062 - val_loss: 0.1576 - val_accuracy: 0.9276

Epoch 63/100
31/31 [==============================] - ETA: 0s - loss: 0.2093 - accuracy: 0.9098
Epoch 00063: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 105ms/step - loss: 0.2093 - accuracy: 0.9098 - val_loss: 0.1551 - val_accuracy: 0.9276

Epoch 64/100
31/31 [==============================] - ETA: 0s - loss: 0.2189 - accuracy: 0.9027
Epoch 00064: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 97ms/step - loss: 0.2189 - accuracy: 0.9027 - val_loss: 0.1562 - val_accuracy: 0.9256

Epoch 65/100
31/31 [==============================] - ETA: 0s - loss: 0.2238 - accuracy: 0.9007
Epoch 00065: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 96ms/step - loss: 0.2238 - accuracy: 0.9007 - val_loss: 0.1759 - val_accuracy: 0.9195

Epoch 66/100
31/31 [==============================] - ETA: 0s - loss: 0.2208 - accuracy: 0.9052
Epoch 00066: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 96ms/step - loss: 0.2208 - accuracy: 0.9052 - val_loss: 0.1698 - val_accuracy: 0.9235

Epoch 67/100
31/31 [==============================] - ETA: 0s - loss: 0.2249 - accuracy: 0.9032
Epoch 00067: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 105ms/step - loss: 0.2249 - accuracy: 0.9032 - val_loss: 0.1603 - val_accuracy: 0.9195

Epoch 68/100
31/31 [==============================] - ETA: 0s - loss: 0.2235 - accuracy: 0.9088
Epoch 00068: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 95ms/step - loss: 0.2235 - accuracy: 0.9088 - val_loss: 0.1577 - val_accuracy: 0.9195

Epoch 69/100
31/31 [==============================] - ETA: 0s - loss: 0.2408 - accuracy: 0.8936
Epoch 00069: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 95ms/step - loss: 0.2408 - accuracy: 0.8936 - val_loss: 0.1762 - val_accuracy: 0.9235

Epoch 70/100
31/31 [==============================] - ETA: 0s - loss: 0.2178 - accuracy: 0.9047
Epoch 00070: val_loss did not improve from 0.15149
31/31 [==============================] - 4s 124ms/step - loss: 0.2178 - accuracy: 0.9047 - val_loss: 0.1613 - val_accuracy: 0.9235

Epoch 71/100
31/31 [==============================] - ETA: 0s - loss: 0.2223 - accuracy: 0.8987
Epoch 00071: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 96ms/step - loss: 0.2223 - accuracy: 0.8987 - val_loss: 0.1549 - val_accuracy: 0.9235

Epoch 72/100
31/31 [==============================] - ETA: 0s - loss: 0.2172 - accuracy: 0.9093
Epoch 00072: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 96ms/step - loss: 0.2172 - accuracy: 0.9093 - val_loss: 0.1606 - val_accuracy: 0.9235

Epoch 73/100
31/31 [==============================] - ETA: 0s - loss: 0.2119 - accuracy: 0.9103
Epoch 00073: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 102ms/step - loss: 0.2119 - accuracy: 0.9103 - val_loss: 0.1853 - val_accuracy: 0.9095

Epoch 74/100
31/31 [==============================] - ETA: 0s - loss: 0.2138 - accuracy: 0.9214
Epoch 00074: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 95ms/step - loss: 0.2138 - accuracy: 0.9214 - val_loss: 0.2074 - val_accuracy: 0.8974

Epoch 75/100
31/31 [==============================] - ETA: 0s - loss: 0.2214 - accuracy: 0.9017
Epoch 00075: val_loss did not improve from 0.15149
31/31 [==============================] - 3s 95ms/step - loss: 0.2214 - accuracy: 0.9017 - val_loss: 0.1758 - val_accuracy: 0.9215

Epoch 76/100
31/31 [==============================] - ETA: 0s - loss: 0.2047 - accuracy: 0.9123
Epoch 00076: val_loss improved from 0.15149 to 0.15097, saving model to model.h5
31/31 [==============================] - 4s 133ms/step - loss: 0.2047 - accuracy: 0.9123 - val_loss: 0.1510 - val_accuracy: 0.9316

Epoch 77/100
31/31 [==============================] - ETA: 0s - loss: 0.2146 - accuracy: 0.9108
Epoch 00077: val_loss improved from 0.15097 to 0.14935, saving model to model.h5
31/31 [==============================] - 5s 161ms/step - loss: 0.2146 - accuracy: 0.9108 - val_loss: 0.1494 - val_accuracy: 0.9356

Epoch 78/100
31/31 [==============================] - ETA: 0s - loss: 0.2225 - accuracy: 0.9093
Epoch 00078: val_loss improved from 0.14935 to 0.14915, saving model to model.h5
31/31 [==============================] - 5s 155ms/step - loss: 0.2225 - accuracy: 0.9093 - val_loss: 0.1491 - val_accuracy: 0.9316

Epoch 79/100
31/31 [==============================] - ETA: 0s - loss: 0.2233 - accuracy: 0.9098
Epoch 00079: val_loss improved from 0.14915 to 0.14742, saving model to model.h5
31/31 [==============================] - 5s 146ms/step - loss: 0.2233 - accuracy: 0.9098 - val_loss: 0.1474 - val_accuracy: 0.9316

Epoch 80/100
31/31 [==============================] - ETA: 0s - loss: 0.2184 - accuracy: 0.9148
Epoch 00080: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 95ms/step - loss: 0.2184 - accuracy: 0.9148 - val_loss: 0.1491 - val_accuracy: 0.9336

Epoch 81/100
31/31 [==============================] - ETA: 0s - loss: 0.2088 - accuracy: 0.9098
Epoch 00081: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 97ms/step - loss: 0.2088 - accuracy: 0.9098 - val_loss: 0.1577 - val_accuracy: 0.9296

Epoch 82/100
31/31 [==============================] - ETA: 0s - loss: 0.2094 - accuracy: 0.9108
Epoch 00082: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 104ms/step - loss: 0.2094 - accuracy: 0.9108 - val_loss: 0.1701 - val_accuracy: 0.9256

Epoch 83/100
31/31 [==============================] - ETA: 0s - loss: 0.2125 - accuracy: 0.9088
Epoch 00083: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 96ms/step - loss: 0.2125 - accuracy: 0.9088 - val_loss: 0.1661 - val_accuracy: 0.9256

Epoch 84/100
31/31 [==============================] - ETA: 0s - loss: 0.2337 - accuracy: 0.9022
Epoch 00084: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 95ms/step - loss: 0.2337 - accuracy: 0.9022 - val_loss: 0.1490 - val_accuracy: 0.9316

Epoch 85/100
31/31 [==============================] - ETA: 0s - loss: 0.2046 - accuracy: 0.9093
Epoch 00085: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 104ms/step - loss: 0.2046 - accuracy: 0.9093 - val_loss: 0.1485 - val_accuracy: 0.9356

Epoch 86/100
31/31 [==============================] - ETA: 0s - loss: 0.2055 - accuracy: 0.9098
Epoch 00086: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 95ms/step - loss: 0.2055 - accuracy: 0.9098 - val_loss: 0.1571 - val_accuracy: 0.9296

Epoch 87/100
31/31 [==============================] - ETA: 0s - loss: 0.2291 - accuracy: 0.9002
Epoch 00087: val_loss did not improve from 0.14742
31/31 [==============================] - 3s 96ms/step - loss: 0.2291 - accuracy: 0.9002 - val_loss: 0.1493 - val_accuracy: 0.9296

Epoch 88/100
31/31 [==============================] - ETA: 0s - loss: 0.2271 - accuracy: 0.9057
Epoch 00088: val_loss improved from 0.14742 to 0.14703, saving model to model.h5
31/31 [==============================] - 5s 159ms/step - loss: 0.2271 - accuracy: 0.9057 - val_loss: 0.1470 - val_accuracy: 0.9296

Epoch 89/100
31/31 [==============================] - ETA: 0s - loss: 0.2251 - accuracy: 0.9057
Epoch 00089: val_loss improved from 0.14703 to 0.14538, saving model to model.h5
31/31 [==============================] - 5s 165ms/step - loss: 0.2251 - accuracy: 0.9057 - val_loss: 0.1454 - val_accuracy: 0.9356

Epoch 90/100
31/31 [==============================] - ETA: 0s - loss: 0.2303 - accuracy: 0.9047
Epoch 00090: val_loss improved from 0.14538 to 0.14396, saving model to model.h5
31/31 [==============================] - 5s 146ms/step - loss: 0.2303 - accuracy: 0.9047 - val_loss: 0.1440 - val_accuracy: 0.9316

Epoch 91/100
31/31 [==============================] - ETA: 0s - loss: 0.2213 - accuracy: 0.9078
Epoch 00091: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 96ms/step - loss: 0.2213 - accuracy: 0.9078 - val_loss: 0.1444 - val_accuracy: 0.9316

Epoch 92/100
31/31 [==============================] - ETA: 0s - loss: 0.2071 - accuracy: 0.9118
Epoch 00092: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 97ms/step - loss: 0.2071 - accuracy: 0.9118 - val_loss: 0.1480 - val_accuracy: 0.9336

Epoch 93/100
31/31 [==============================] - ETA: 0s - loss: 0.2279 - accuracy: 0.9123
Epoch 00093: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 95ms/step - loss: 0.2279 - accuracy: 0.9123 - val_loss: 0.1462 - val_accuracy: 0.9336

Epoch 94/100
31/31 [==============================] - ETA: 0s - loss: 0.1996 - accuracy: 0.9183
Epoch 00094: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 106ms/step - loss: 0.1996 - accuracy: 0.9183 - val_loss: 0.1707 - val_accuracy: 0.9235

Epoch 95/100
31/31 [==============================] - ETA: 0s - loss: 0.2098 - accuracy: 0.9143
Epoch 00095: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 95ms/step - loss: 0.2098 - accuracy: 0.9143 - val_loss: 0.1634 - val_accuracy: 0.9276

Epoch 96/100
31/31 [==============================] - ETA: 0s - loss: 0.2165 - accuracy: 0.9073
Epoch 00096: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 106ms/step - loss: 0.2165 - accuracy: 0.9073 - val_loss: 0.1547 - val_accuracy: 0.9235

Epoch 97/100
31/31 [==============================] - ETA: 0s - loss: 0.2225 - accuracy: 0.9098
Epoch 00097: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 111ms/step - loss: 0.2225 - accuracy: 0.9098 - val_loss: 0.1605 - val_accuracy: 0.9276

Epoch 98/100
31/31 [==============================] - ETA: 0s - loss: 0.2195 - accuracy: 0.9098
Epoch 00098: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 105ms/step - loss: 0.2195 - accuracy: 0.9098 - val_loss: 0.1502 - val_accuracy: 0.9316

Epoch 99/100
31/31 [==============================] - ETA: 0s - loss: 0.2082 - accuracy: 0.9073
Epoch 00099: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 95ms/step - loss: 0.2082 - accuracy: 0.9073 - val_loss: 0.1447 - val_accuracy: 0.9336

Epoch 100/100
31/31 [==============================] - ETA: 0s - loss: 0.2142 - accuracy: 0.9108
Epoch 00100: val_loss did not improve from 0.14396
31/31 [==============================] - 3s 98ms/step - loss: 0.2142 - accuracy: 0.9108 - val_loss: 0.1445 - val_accuracy: 0.9376

```
Y_pred = model.predict(X_val)

Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Purples, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/d4dd4119-f77f-4645-a380-64a342cdff9f)

```
# accuracy plot 
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/5e37b8b3-c7e1-458c-aeca-501db54686dd)

> 처음부터 좋은 결과인 이유는 모델을 실수로 2번 돌린 코드이기 때문

```
from skimage import io
from keras.preprocessing import image
img = image.load_img('../input/tifpng-dataset/ID_0204_Z_0066.png', grayscale=False, target_size=(64, 64))
show_img=image.load_img('../input/tifpng-dataset/ID_0204_Z_0066.png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
print(img)
x = np.expand_dims(x, axis = 0)
x /= 255


custom = model.predict(x)
print(custom)
print(custom[0])

plt.imshow(show_img)
plt.show()

a=custom[0]
ind=np.argmax(a)
print(ind)

print('Prediction:',disease_class[ind])
```

![image](https://github.com/UGeunJi/ResNet50_COVID19_Diagosis/assets/84713532/8296d271-c2ee-4879-8c41-83f9d679d7cb)

```
pix = np.array(img)
print(img.size)
print(pix)
```

(64, 64)
[[[245 245 245]
  [245 245 245]
  [245 245 245]
  ...
  [246 246 246]
  [246 246 246]
  [246 246 246]]

 [[245 245 245]
  [246 246 246]
  [246 246 246]
  ...
  [245 245 245]
  [245 245 245]
  [245 245 245]]

 [[245 245 245]
  [245 245 245]
  [246 246 246]
  ...
  [245 245 245]
  [245 245 245]
  [245 245 245]]

 ...

 [[246 246 246]
  [245 245 245]
  [246 246 246]
  ...
  [245 245 245]
  [245 245 245]
  [245 245 245]]

 [[245 245 245]
  [245 245 245]
  [246 246 246]
  ...
  [245 245 245]
  [245 245 245]
  [245 245 245]]

 [[245 245 245]
  [245 245 245]
  [246 246 246]
  ...
  [245 245 245]
  [245 245 245]
  [245 245 245]]]
