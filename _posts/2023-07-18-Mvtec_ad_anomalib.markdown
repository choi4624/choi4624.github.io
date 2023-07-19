---
layout: post
title: "2023-07-18-Mvtec_ad_anomalib"
date: 2023-07-18 10:06:00 +0900
categories: computer
tags: [python, ipynb, data,kaggle,swproject,pytorch,cuDNN]
---
# MVTEC Anomaliy detection 을 진행하기 위한 anomalib 실행 과정

## 개요

mvtec-ad는 가장 유용해서인지 이상 탐지 예시 모델로 굉장히 많이 사용되는 데이터셋이다.
기본적으로 cnn을 통해 직접 이미지를 바탕으로 good 과 bad(및 기타 등등)을 분류하기 때문에 라벨이 있는 다른 데이터와 다르게 직접 cnn 기반 이미지 인식을 통해 학습을 진행하는 것이 좋은 데이터다.
정확히 말하면 데이터셋에 대한 설정은 명시되어 있지 않으나, 폴더 구조를 통해 라벨링이 가능한 성격의 데이터로, 이 데이터를 전처리 하는 것도 중요한 과정 중에 하나이다.

## 기본 cnn MVTEC-AD

anomalib를 통한 학습을 진행하기 이전에, 기본적인 cnn 모델을 통해 MVTEC을 학습하고 데이터 전처리 과정에 대한 간단한 학습 절차를 진행하고자 한다.

데이터 구조는 아래와 같으니 이와 유사한 데이터에도 전처리 절차를 진행해도 무방하다.

```
#MVTec-Anomaly-Detection
      #data
        #├── class1
        # │   ├── test
        # │   │   ├── good
        # │   │   ├── defect
        # │   └── train
        # │       └── good
        # ├── class2
        # │   ├── test
        # │   │   ├── good
        # │   │   ├── defect
        # │   └── train
        # │       └── good
        # ...
```

### 기본 cnn 모델 전처리 과정

```
import os
from keras.preprocessing.image import ImageDataGenerator
print(os.getcwd())
os.getcwd()

# Define directories for your dataset
base_dir = './mvtec_anomaly_detection/'

train_dir = os.path.join(base_dir, 'pill', 'train')
validation_dir = os.path.join(base_dir, 'pill', 'test')
test_dir = os.path.join(base_dir, 'pill', 'test')


# Define image preprocessing and augmentation options
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for test set

# Generate batches of augmented training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to a fixed size
    batch_size=32,
    class_mode='categorical'  # For classification tasks
)

# Generate batches of validation data
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Generate batches of test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

```

기본적인 cnn 전처리 코드로 원 데이터 셋이 train / test 로 나뉜 만큼 validation 데이터의 경우 test 데이터에서 가져다가 사용한다.

```

import tensorflow as tf
from tensorflow import keras
import numpy as np

```

그 다음 cnn 모델 중 rasnet 50을 사용하기 위해 keras에 정의된 rasnet을 사용한다.

아래는 rasnet 사용 전 기본적인 cnn 모델의 경우 이러한 방식으로 모델을 정의하고 사용할 수 있는 방법 중 하나이다. MNIST 글자 인식 같으면 이정도면 충분한 모델이나, MVTEC 데이터는 절대 그렇지 않으므로 더 복잡한 모델인 rasnet으로 시작한다.

```
# Create a sequential model
model = Sequential()

# Add convolutional layers

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Assuming 2 classes (good and defect)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
```

그 다음에 rasnet을 적용하여 학습을 진행해본다

```
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=8,
    **kwargs
)

model = ResNet50(weights='imagenet')
```

```

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
```

### 결과

---

Epoch 2/20
9/9 [==============================] - 15s 2s/step - loss: 41643536384.0000 - accuracy: 0.0000e+00 - val_loss: 4743202304.0000 - val_accuracy: 0.0539
Epoch 3/20
9/9 [==============================] - 20s 2s/step - loss: 44334411776.0000 - accuracy: 0.1610 - val_loss: 5175751680.0000 - val_accuracy: 0.1257
Epoch 4/20
9/9 [==============================] - 15s 2s/step - loss: 47982247936.0000 - accuracy: 0.2397 - val_loss: 6002683392.0000 - val_accuracy: 0.1138
Epoch 5/20
9/9 [==============================] - 15s 2s/step - loss: 54250090496.0000 - accuracy: 0.0000e+00 - val_loss: 6354566144.0000 - val_accuracy: 0.1437
Epoch 6/20
9/9 [==============================] - 15s 2s/step - loss: 58322538496.0000 - accuracy: 0.1610 - val_loss: 6992474624.0000 - val_accuracy: 0.1557
Epoch 7/20
9/9 [==============================] - 15s 2s/step - loss: 66296102912.0000 - accuracy: 0.2397 - val_loss: 8333232640.0000 - val_accuracy: 0.1437
Epoch 8/20
9/9 [==============================] - 15s 2s/step - loss: 72777580544.0000 - accuracy: 0.0000e+00 - val_loss: 8866929664.0000 - val_accuracy: 0.1138
Epoch 9/20
9/9 [==============================] - 15s 2s/step - loss: 71693713408.0000 - accuracy: 0.2397 - val_loss: 8689728512.0000 - val_accuracy: 0.1437
Epoch 10/20
9/9 [==============================] - 15s 2s/step - loss: 79761481728.0000 - accuracy: 0.1199 - val_loss: 8933327872.0000 - val_accuracy: 0.1497
Epoch 11/20
9/9 [==============================] - 15s 2s/step - loss: 80549502976.0000 - accuracy: 0.1199 - val_loss: 8969754624.0000 - val_accuracy: 0.1437
Epoch 12/20
9/9 [==============================] - 15s 2s/step - loss: 86114844672.0000 - accuracy: 0.2397 - val_loss: 10996793344.0000 - val_accuracy: 0.1557
Epoch 13/20
9/9 [==============================] - 15s 2s/step - loss: 90812456960.0000 - accuracy: 0.0000e+00 - val_loss: 11797346304.0000 - val_accuracy: 0.1497
Epoch 14/20
9/9 [==============================] - 15s 2s/step - loss: 93791313920.0000 - accuracy: 0.2397 - val_loss: 10306733056.0000 - val_accuracy: 0.1437
Epoch 15/20
9/9 [==============================] - 15s 2s/step - loss: 78944337920.0000 - accuracy: 0.1199 - val_loss: 10068012032.0000 - val_accuracy: 0.1557
Epoch 16/20
9/9 [==============================] - 15s 2s/step - loss: 85686607872.0000 - accuracy: 0.2397 - val_loss: 11911929856.0000 - val_accuracy: 0.1138
Epoch 17/20
9/9 [==============================] - 15s 2s/step - loss: 64439443456.0000 - accuracy: 0.1199 - val_loss: 7003056128.0000 - val_accuracy: 0.1557
Epoch 18/20
9/9 [==============================] - 15s 2s/step - loss: 41878708224.0000 - accuracy: 0.1199 - val_loss: 5362637312.0000 - val_accuracy: 0.1557
Epoch 19/20
9/9 [==============================] - 15s 2s/step - loss: 62394286080.0000 - accuracy: 0.0412 - val_loss: 10939133952.0000 - val_accuracy: 0.1557
Epoch 20/20
9/9 [==============================] - 15s 2s/step - loss: 81569595392.0000 - accuracy: 0.1199 - val_loss: 13032003584.0000 - val_accuracy: 0.1497

---

플롯을 그리는게 의미가 없을 정도로 정확도가 낮으며, 이 상태로는 테스트의 의미도 없어졌다.

더 나은 모델을 사용해야 할 필요가 있는데, 더 나은 모델 중 pytorch 기반의 anomalib가 다양한 모델을 지원하면서 정확도가 꽤나 괜찮은 모델을 보여준다. (오버피팅이라도 난 것 마냥 높음)

[openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. (github.com)](https://github.com/openvinotoolkit/anomalib)

* 왜 anomalib를 사용하나

1. anomalib 의 사용하기 비교적 간편한 코드
2. 학습이 가능한 하드웨어 스펙 (GPU VRAM 사용이 6GB내로 억제됨)
3. 굉장히 높은 정확도
4. colab과 연동해서 사용할 수 있음

이 4 가지 이유가 colab 환경에서 anomalib를 사용하여 프로젝트를 진행하게 된 이유다.

아래 내용부터는 코드와 함께 사용에 대한 설명을 추가하여 진행한다.

## anomalib 사용

### 구글 드라이브 연동

```
from google.colab import drive
drive.mount('/content/drive')

```

* 구글 드라이브를 마운트하고 적당한 경로에 anomalib를 다운로드 하고 마운트 한 다음 pip 를 해당 경로에서 진행한다.
* git 다운시[https://github.com/openvinotoolkit/anomalib]() 로 접속하여 main(master) 브랜치를 다운로드 한 다음 드랑이브에 업로드 한다.
* 굳이 그래야 하는 이유는 이거 pip 생각보다 무거운데다, pip 안에 있는걸 그대로 쓰지 않고 해당 라이브러리 내에 있는 모델 config를 그대로 사용할 예정이기 때문이다.
  (MVTEC-AD 용 최적화 컨피그가 이미 있다.)

`%cd /content/drive/MyDrive/colab/anomalib/anomalib-main`

예시 경로

`%pip install -e . -q`

해당 경로에서 cd 한 다음 위의 방법으로 노트북에서 pip install을 진행한 다음 정상적으로 설치되었다면 `%pip list` 를 진행하여 설치가 정상적으로 되었는지를 확인한다.

본 예제에선 몇 가지 추가 패키지가 필요하므로, 해당 패키지까지 같이 설치한다.

`%pip install openvino`

`%pip install wandb`

### 학습 데이터 설정 

![](../assets/20230719_105233_2023-07-19_093944.png)

학습 데이터 역시 구글 드라이브에 설정하여 관련 데이터를 집어넣어 colab 런타임에 매번 데이터를 넣지 않도록 구글 드라이브 마운트시 바로 가져올 수 있도록 한다. 

MVTEC 데이터가 대략 5GB 정도 되므로, 학교용 계정으로는 안하는게 바람직하고 개인 계정으로 사용하거나 임시로 계정을 만들어서 코랩용도로만 사용해도 괜찮다. 

주의사항: colab에 드라이브를 마운트하면 드라이브 데이터를 조작하면 드라이브에 바로 반영되니 데이터 삭제 및 유실에 주의 

* 공유된 링크로 받는 것은 서로 다른 계정간 연동 문제로 인해 편집 권한이 있어도 정상 작동하지 않을 가능성이 높다.


### 패키지 설정

커널 재시작을 한 다음 (연결 끊지 말고) 아래의 import를 진행하여 코드 실행을 준비한다. 어떤 경고도 허용되지 않고 import도 다 정상적으로 되어야 한다.

코랩이 아닌 다른 환경에서 한다면, import 오류가 나는 목록들은 다 import 해주어야 하므로 참고

```
import numpy as np
import matplotlib.pyplot as plt
import os, pprint, yaml, warnings, math, glob, cv2, random, logging

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')
logger = logging.getLogger("anomalib")

import anomalib
from pytorch_lightning import Trainer, seed_everything
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
```

### gpu setting

해당 코드 이후에 gpu memory가 보여야 합니다.

```
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
```

### 모델 경로 설정 및 모델 설정 상태 확인

anomalib에 있는 config들을 가져다가 사용하기 위한 작업으로, 기본값은 patchcore다. 성능 좋다.

```
CONFIG_PATHS = '/content/drive/MyDrive/colab/anomalib/anomalib-main/src/anomalib/models'
MODEL_CONFIG_PAIRS = {
    'patchcore': f'{CONFIG_PATHS}/patchcore/config.yaml',
    'padim':     f'{CONFIG_PATHS}/padim/config.yaml',
    'cflow':     f'{CONFIG_PATHS}/cflow/config.yaml',
    'dfkde':     f'{CONFIG_PATHS}/dfkde/config.yaml',
    'dfm':       f'{CONFIG_PATHS}/dfm/config.yaml',
    'ganomaly':  f'{CONFIG_PATHS}/ganomaly/config.yaml',
    'stfpm':     f'{CONFIG_PATHS}/stfpm/config.yaml',
    'fastflow':  f'{CONFIG_PATHS}/fastflow/config.yaml',
    'draem':     f'{CONFIG_PATHS}/draem/config.yaml',
    'reverse_distillation': f'{CONFIG_PATHS}/reverse_distillation/config.yaml',
}
```

```
MODEL = 'patchcore' # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
## patchcore 가 좋은것 같아요
print(open(os.path.join(MODEL_CONFIG_PAIRS[MODEL]), 'r').read())
```

위 코드를 실행하면 정상적인 경우 다음과 같은 설정 결과가 나와야 한다. 설정을 하나씩 읽어보는 것을 추천한다.

어떤 작업을 하고 어떤 카테고리랑 배치 사이즈랑 이미지 관련 설정까지 전부 위 config 안에 들어가 있는 것이다.

---

dataset:
name: mvtec
format: mvtec ## MVTec 형식으로 데이터 설정
path: [./datasets/MVTec](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/choi4/Downloads/datasets/MVTec)
task: segmentation ## 분류
category: bottle ## 병으로 카테고리화 (기본값)
train_batch_size: 32 ## 배치 사이즈 - 메모리에 직접적으로 영향 줌
eval_batch_size: 32
num_workers: 8
image_size: 256 # dimensions to which images are resized (mandatory) ## 이미지 크기 (resize)

center_crop: 224 # dimensions to which images are center-cropped after resizing (optional)
normalization: imagenet # data distribution to which the images will be normalized: [none, imagenet]
transform_config:
train: null
eval: null
test_split_mode: from_dir # options: [from_dir, synthetic]

// 테스트랑 검증 비율 설정해놓는 설정
test_split_ratio: 0.2 # fraction of train images held out testing (usage depends on test_split_mode)
val_split_mode: same_as_test # options: [same_as_test, from_test, synthetic]
val_split_ratio: 0.5 # fraction of train/test images held out for validation (usage depends on val_split_mode)
tiling:
apply: false
tile_size: null
stride: null ## 포폭이 없다?
remove_border_count: 0
use_random_tiling: False

...

---

## 데이터 설정 및 업데이트

```
## 대상이 될 데이터 새로 설정

new_update = {
    "path": '/content/drive/MyDrive/colab/mvtec_anomaly_detection',
    'category': 'screw',
    'image_size': 256,
    'train_batch_size':48,
    'seed': 101
}
```

데이터를 새로 설정하면서 카테고리랑 배치 사이즈를 바꾸고 진행한다. seed 값은 랜덤화된 데이터를 일정 수준으로 고정시켜 주는 역할을 한다. 매번 새로운 데이터가 나와 학습에 혼란을 주는 일이 없도록 막을 수 있다. seed를 바꿔서 하는 것도 방법이므로 바꿨을 때랑 안바꿀 때의 차이를 비교하는 것도 중요한 요소일 듯 하다.

```
# update yaml key's value
def update_yaml(old_yaml, new_yaml, new_update):
    # load yaml
    with open(old_yaml) as f:
        old = yaml.safe_load(f)

    temp = []
    def set_state(old, key, value):
        if isinstance(old, dict):
            for k, v in old.items():
                if k == 'project':
                    temp.append(k)
                if k == key:
                    if temp and k == 'path':
                        # right now, we don't wanna change `project.path`
                        continue
                    old[k] = value
                elif isinstance(v, dict):
                    set_state(v, key, value)

    # iterate over the new update key-value pari
    for key, value in new_update.items():
        set_state(old, key, value)

    # save the updated / modified yaml file
    with open(new_yaml, 'w') as f:
        yaml.safe_dump(old, f, default_flow_style=False)
```

```
# let's set a new path location of new config file
new_yaml_path = CONFIG_PATHS + '/' + list(MODEL_CONFIG_PAIRS.keys())[0] + '_new.yaml'

# run the update yaml method to update desired key's values
update_yaml(MODEL_CONFIG_PAIRS[MODEL], new_yaml_path, new_update)
```

데이터 재설정 이후 다시 출력하여 정상적으로 반영되었는지 확인한다.

```
with open(new_yaml_path) as f:
    updated_config = yaml.safe_load(f)
pprint.pprint(updated_config) # check if it's updated
```

---

#### SCREW 데이터에 대한 설정

'dataset': {'category': 'screw',
'center_crop': 224,
'eval_batch_size': 32,
'format': 'mvtec',
'image_size': 256,
'name': 'mvtec',
'normalization': 'imagenet',
'num_workers': 8,
'path': '/content/drive/MyDrive/colab/mvtec_anomaly_detection',
'task': 'segmentation',
'test_split_mode': 'from_dir',
'test_split_ratio': 0.2,
'tiling': {'apply': False,
'random_tile_count': 16,
'remove_border_count': 0,
'stride': None,
'tile_size': None,
'use_random_tiling': False},
'train_batch_size': 48,
'transform_config': {'eval': None, 'train': None},
'val_split_mode': 'same_as_test',
'val_split_ratio': 0.5},
'logging': {'log_graph': False, 'logger': []},
'metrics': {'image': ['F1Score', 'AUROC'],
'pixel': ['F1Score', 'AUROC'],...

```
               'log_images': True,
               'mode': 'full',
               'save_images': True,
               'show_images': False}}
```

이건 이미지 로깅에 대한 것인데, 이거 저장해야 결과를 볼 수 있으므로 반드시 설정

---

## 모델 학습 시작

```
if updated_config['project']['seed'] != 0:
    print(updated_config['project']['seed'])
    seed_everything(updated_config['project']['seed'])
```

모델이 잘 적용되어 있는지 확인하고 로깅도 할 수 있도록 설정도 진행한다.

```
# It will return the configurable parameters in DictConfig object.
config = get_configurable_parameters(
    model_name=updated_config['model']['name'],
    config_path=new_yaml_path
)
```

```
# pass the config file to model, logger, callbacks and datamodule
model      = get_model(config)
experiment_logger = get_experiment_logger(config)
callbacks  = get_callbacks(config)
datamodule = get_datamodule(config)
```

기본이 될 모델들도 다 불러오고 설정도 다 적용했으면 학습을 시작한다. 

* 모델은 외부에서 별도로 가져오는 것이기 때문에 인터넷 환경에 영향을 받으므로 주의

### 학습 진행 

```
# start training
trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
trainer.fit(model=model, datamodule=datamodule)
```

---



INFO:pytorch_lightning.callbacks.model_summary:
| Name                  | Type                     | Params

0 | image_threshold       | AnomalyScoreThreshold    | 0
1 | pixel_threshold       | AnomalyScoreThreshold    | 0
2 | model                 | PatchcoreModel           | 24.9 M
3 | image_metrics         | AnomalibMetricCollection | 0
4 | pixel_metrics         | AnomalibMetricCollection | 0
5 | normalization_metrics | MinMax                   | 0

24.9 M    Trainable params
0         Non-trainable params
24.9 M    Total params
99.450    Total estimated model params size (MB)

---


![](../assets/20230719_105727_image.png)

1회 epoch가 꽤 긴 시간이 걸리는 무거운 모델이라, 모델을 한 번 학습하고 진행한다. 

만약 여러 번 학습을 하고 싶다면 config에 아래 값들을 수정한다. new_update 부분에서 '키' : 값 형식으로 지정한다음 업데이트 하면 된다. 

```
  'gradient_clip_algorithm': 'norm',
             'gradient_clip_val': 0,
             'limit_predict_batches': 1.0,
             'limit_test_batches': 1.0,
             'limit_train_batches': 1.0,
             'limit_val_batches': 1.0,
             'log_every_n_steps': 50,
****** 'max_epochs': 1, ******
             'max_steps': -1,
```

참고: `'precision': 32,` 값도 수정할 수 있는데, gpu 가속기라면 16으로 해서 빠른 속도의 연산과 약간의 정확도 손실을 기대할 수 있다. 

![](../assets/20230719_110723_image.png)

대부분의 최신 gpu의 경우 FP16에 대한 가속을 보장하므로, 거대한 모델이라면 정확도를 희생해 가속을 기대할 수도 있다.

## best model check + test 

```
# load best model from checkpoint before evaluating
load_model_callback = LoadModelCallback(
    weights_path=trainer.checkpoint_callback.best_model_path
)
trainer.callbacks.insert(0, load_model_callback)
trainer.test(model=model, datamodule=datamodule)
```


![](../assets/20230719_110917_image.png)

epoch가 여러 개라면 가장 좋은 모델에 대한 확인과 테스트 결과값을 낸다. 

## 시각화 

```
RESULT_PATH = os.path.join(
    updated_config['project']['path'],
    updated_config['model']['name'],
    updated_config['dataset']['format'],
    updated_config['dataset']['category']
)
RESULT_PATH
```

여태까지 설정해놓은 프로젝트, 모델, 데이터, 카테고리를 바탕으로 모델을 저장했고 또 이를 불러와서 시각화를 진행한다. 

### 시각화 정의 

```
# a simple function to visualize the model's prediction (anomaly heatmap)
def vis(paths, n_images, is_random=True, figsize=(16, 16)):
    for i in range(n_images):
        image_name = paths[i]
        if is_random: image_name = random.choice(paths)
        img = cv2.imread(image_name)[:,:,::-1]

        category_type = image_name.split('/')[-4:-3:][0]
        defected_type = image_name.split('/')[-2:-1:][0]

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(
            f"Category : {category_type} and Defected Type : {defected_type}",
            fontdict={'fontsize': 20, 'fontweight': 'medium'}
        )
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
```

opencv기반 라이브러리로 이미지를 불러오고 시각화를 반영한다. 

```
for content in os.listdir(RESULT_PATH):
    #if content == 'images': 작동 안할경우 주석처리 지우기 
        full_path = glob.glob(os.path.join(RESULT_PATH, content, '**',  '*.png'), recursive=True)
        print('Total Image ', len(full_path))
        print(full_path[0].split('/'))
        print(full_path[0].split('/')[-2:-1:])
        print(full_path[0].split('/')[-4:-3:])
```

아래와 같이 결과가 나오면 성공 

```
Total Image  162
['.', 'results', 'patchcore', 'mvtec', 'screw', 'run', 'images', 'image_ROC.png']
['images']
['screw']
```

```

def get_all_images():
    for content in os.listdir(RESULT_PATH):
        if content == 'images':
            full_paths += glob.glob(os.path.join(RESULT_PATH, content, '**', '*.png'), recursive=True)
            print('Total Image ', len(full_path))
            print(full_path[0].split('/'))
            print(full_path[0].split('/')[-2:-1:])
            print(full_path[0].split('/')[-4:-3:])



```

`full_paths = get_all_images()`를 실행하여 경로에 있는 이미지들을 가져오고 결과를 출력하도록 한다. 

`vis(full_path, 10, is_random=True, figsize=(30, 30))` 를 실행해 다음과 같은 이미지가 나오면 시각화 완료 


![](../assets/20230719_111317_image.png)

## 가중치 및 결과 이미지 저장 

```
import shutil
shutil.make_archive('results-anmalib-screw', 'zip', '/content/results')
```

`!cp -r '/content/results-anmalib-screw.zip' /content/drive/MyDrive/colab/results` 

경로는 환경에 따라 다르므로, 적절한 경로르 지정하고 드라이브에 꼭 옮기도록 한다. 


## 결론

1. 모델 잘 만들어서 논문 받는 사람이 많으니 모델은 가져다 쓰세요.. 제발...
2. epoch수를 늘리면 오히려 과적합이 날 기세
3. 사용한 patchcore 라는 모델은[[2106.08265] Towards Total Recall in Industrial Anomaly Detection (arxiv.org)](https://arxiv.org/abs/2106.08265) 해당 논문을 통해 만들어진 모델로 기본적으로 인코더 - 디코더 기반의 cnn 모델로 해석된다.


![](../assets/20230719_111837_image.png)

다만 손상된 부분에 대한 feature를 매우 큰 메모리 뱅크에 넣은 다음 여기에 대한 Nearest Neighbour Search를 진행하여 이상 탐지를 이미지 바탕으로 진행하는 방식을 사용하는 점이 다른 모델과의 차이가 있다. 또한, unsupervised training으로 patch를 탐색하고 segmentation을 진행하는 것이 이 모델의 특징이라 볼 수 있다. 


4. 집에 있는 인텔 arc gpu로 pytorch를 하려 했는데 최근에 파이토치가 2.x 버전으로 올라가면서 인텔 pytorch extension과의 호환이 어긋나버려 진행이 안된다. 아무리 해도 관련 라이브러리들이 다 2.x 기반 파이토치를 쓰게 되어서 실패!
5. 또한 해당 모델을 바탕으로 베포하는 작업을 진행해야 하는데 베포 작업은 이전에 했던 오만가지 트레이닝과는 아예 인연이 없는데다 spring으로만 웹을 만든 사람에게 너무 무서운 요소가 될 것 같다.
