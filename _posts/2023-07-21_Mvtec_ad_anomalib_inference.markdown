---
%pip install "fastapi[all]"
---
# anomalib 를 이용한 MVTEC-AD 데이터셋의 웹 서비스 베포 과정

---

**Contents :**

* [모델편성](#모델편성)
* [로컬 환경 진행](#로컬)
* [문제](#문제)
* [라이브러리 고치기](#라이브러리수정)
* [추론 진행](#추론진행)
* [공개 배포](#공개배포)

---

본 프로젝트에서는 이전에 코랩 환경에서 진행했던 Mvtec AD 학습 프로젝트를 통해 생성한 모델을 바탕으로 추론 과정을 만들고 베포하는 작업을 진행한다.
예를들어 rs.i4624.tk/predict 등의 웹 페이지로 연결해서 이미지를 업로드 하면, 업로드 된 이미지를 추론하여 이상 영역을 확인하고 (필요하다면) 분류도 진행할 수 있도록 한다.

필요하다면 해당 코드와 같이 확인하여 동작을 보는 것도 추천한다.

[SWbootProject_2023-7/2nd_cnn/fastapi/api/final.py at main · choi4624/SWbootProject_2023-7 (github.com)](https://github.com/choi4624/SWbootProject_2023-7/blob/main/2nd_cnn/fastapi/api/final.py)

## 모델편성

이전 게시글에서 나온 모델을 업로드하고 해당 모델을 가져와서 추론 절차를 진행하도록 베포할 프로젝트에 첨부한다.

참고로 용량 문제로 인해 github에 모델을 그냥 올릴 수는 없기 때문에, 해당 모델은 가능하면 직접 만들어서 첨부하는 것을 추천한다.

본 프로젝트에서는 api 폴더 아래에 모델과 해당 모델을 학습할 설정 파일을 올려 해당 모델이 원하는 방식으로 실행될 수 있도록 설정한다.

폴더 구조는 아래와 같이 되어 있다.

![](../assets/20230721_131016_2023-07-21_131004.png)

model.ckpt에 .gitignore를 반드시 걸어주자, 안그러면 vscode에서 커밋 잘못 넣어가지고 push가 안된다.

## 시각화

먼저, 해당 프로젝트의 추론 코드를 실행하는 방법을 확인해봐야 하는데, anomalib 내에 있는 tools를 사용하여 추론을 하고 시각화를 내리는 것으로 나왔다. anomalib에서 제공하는 스크립트를 바탕으로 추론을 진행해주는 편이 좋기 때문에 우선 로컬 노트북에서 관련 작업을 진행하였다.

![](../assets/20230721_132158_2023-07-21_132105.png)

어쩌다 보니 웹과 로컬 환경에서 각각 다른 코드를 실행하게 되었는데, 이는 파이썬을 정말 못한 어떤 사람이 오류를 해석하고 이를 해결하는게 너무 빙 돌아가는 것에 잘못 선택까지 해서 그렇다 ㅜㅜ

### 로컬 환경에서 실행하는 추론

주피터노트북에서 실행하는 코드이므로, 관련 코드를 첨부한다.

```python
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

먼저 라이브러리들을 import 한다 이전 게시글과 동일하게 프로젝트 단위로 anomalib를 설치해도 되고, 아니면 그냥 pip install anmoalib 해도 상관은 없다.

```python
%pip install "fastapi[all]"

%pip install openvino
%pip install wandb
%pip install matplotlib
```

똑같이 torch를 import하고 실행하면 되는데, 실행 환경에 따라 gpu를 첨부해도 되고 안해도 된다. gpu 없어도 4개 코어를 좀 혹사시키기만 해도 잘 실행 할 수 있다.

어짜피 이용자 수가 스레드 수보다 많지 않을거 안다.

```python
CONFIG_PATHS = '/{$anomalib installed path}/src/anomalib/models'
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

anomalib 설치 파일로부터 기본 config를 가져온다. 각 알고리즘에 대한 정의를 anomalib에서 진행하기 때문에 필수.

`MODEL = 'patchcore' # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'`

모델을 지정하고

```python
new_update = {
    "path": '/{$source_file_path}/mvtec_anomaly_detection',
    'category': 'hazelnut', 
    'image_size': 256,
    'train_batch_size':16,
    'seed': 101
}
```

새로운 모델을 학습시킬 소스 파일의 경로까지 지정해준다.

카테고리 설정 폴더에 학습할 이미지들이 있으면 된다.

```python
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


# let's set a new path location of new config file 
new_yaml_path = CONFIG_PATHS + '/' + list(MODEL_CONFIG_PAIRS.keys())[0] + '_new.yaml'

# run the update yaml method to update desired key's values
update_yaml(MODEL_CONFIG_PAIRS[MODEL], new_yaml_path, new_update)


```

이전에 했던 대로 config를 다시 반영하여 저장한다.

```python
with open(new_yaml_path) as f:
    updated_config = yaml.safe_load(f)


### working condition  
pprint.pprint(updated_config)
```

```json
{'dataset': {'category': 'hazelnut',
             'center_crop': 224,
             'eval_batch_size': 32,
             'format': 'mvtec',
             'image_size': 256,
             'name': 'mvtec',
             'normalization': 'imagenet',
             'num_workers': 8,
             'path': '/home/i4624/vscode/gitclone/SWbootProject_2023-7/2nd_cnn/content/mvtec_anomaly_detection',
             'task': 'segmentation',
             'test_split_mode': 'from_dir',
             'test_split_ratio': 0.2,
             'tiling': {'apply': False,
                        'random_tile_count': 16,
                        'remove_border_count': 0,
                        'stride': None,
                        'tile_size': None,
                        'use_random_tiling': False},
             'train_batch_size': 16,
             'transform_config': {'eval': None, 'train': None},
             'val_split_mode': 'same_as_test',
             'val_split_ratio': 0.5},
 'logging': {'log_graph': False, 'logger': []},
 'metrics': {'image': ['F1Score', 'AUROC'],
             'pixel': ['F1Score', 'AUROC'],
             'threshold': {'manual_image': None,
                           'manual_pixel': None,
                           'method': 'adaptive'}},
 'model': {'backbone': 'wide_resnet50_2',
           'coreset_sampling_ratio': 0.1,
           'layers': ['layer2', 'layer3'],
           'name': 'patchcore',
           'normalization_method': 'min_max',
           'num_neighbors': 9,
           'pre_trained': True},
 'optimization': {'export_mode': None},
 'project': {'path': './results', 'seed': 101},
 'trainer': {'accelerator': 'auto',
             'accumulate_grad_batches': 1,
             'auto_lr_find': False,
             'auto_scale_batch_size': False,
             'benchmark': False,
             'check_val_every_n_epoch': 1,
             'default_root_dir': None,
             'detect_anomaly': False,
             'deterministic': False,
             'devices': 1,
             'enable_checkpointing': True,
             'enable_model_summary': True,
             'enable_progress_bar': True,
             'fast_dev_run': False,
             'gradient_clip_algorithm': 'norm',
             'gradient_clip_val': 0,
             'limit_predict_batches': 1.0,
             'limit_test_batches': 1.0,
             'limit_train_batches': 1.0,
             'limit_val_batches': 1.0,
             'log_every_n_steps': 50,
             'max_epochs': 1,
             'max_steps': -1,
             'max_time': None,
             'min_epochs': None,
             'min_steps': None,
             'move_metrics_to_cpu': False,
             'multiple_trainloader_mode': 'max_size_cycle',
             'num_nodes': 1,
             'num_sanity_val_steps': 0,
             'overfit_batches': 0.0,
             'plugins': None,
             'precision': 32,
             'profiler': None,
             'reload_dataloaders_every_n_epochs': 0,
             'replace_sampler_ddp': True,
             'strategy': None,
             'sync_batchnorm': False,
             'track_grad_norm': -1,
             'val_check_interval': 1.0},
 'visualization': {'image_save_path': None,
                   'log_images': True,
                   'mode': 'full',
                   'save_images': True,
                   'show_images': False}}
```

잘 저장했으면 config가 잘 되어있는지 출력도 진행

```python
# It will return the configurable parameters in DictConfig object.
config = get_configurable_parameters(
    model_name=updated_config['model']['name'],
    config_path=new_yaml_path
)

from torchvision.datasets.folder import IMG_EXTENSIONS


img_paths = sorted(
        [
            os.path.join(dirpath,filename) 
            for dirpath, _, filenames in os.walk(
                os.path.join(config.dataset.path, config.dataset.category, 'test')
            )
            for filename in filenames if filename.endswith(IMG_EXTENSIONS) 
            and not filename.startswith(".")
        ],
    )

img_paths = [img_path for img_path in img_paths if not "good" in img_path]
img_paths = random.sample(img_paths, 10)
img_paths
```

모델과 이미지 저장소 역시 잘 지정해준다.

img_paths 안에 있는 폴더 안 이미지 중 10개를 임의 샘플로 하여 저장

```python


for infer_img_path in img_paths:
    !python3 /{$ anomalib 경로}/anomalib2/tools/inference/lightning_inference.py --config 
     /{$ anomalib 경로}/anomalib/src/anomalib/models/patchcore/config.yaml    --weights 
     /{$ 모델 저장 경로}/fastapi/results/patchcore/mvtec/hazelnut/run/weights/model.ckpt  
      --input {infer_img_path}  --output ./infer_results      --visualization_mode "full"
```

이제 config와 weight를 가져다가 그대로 추론을 진행한 다음 ./infer_results 폴더에 알아서 집어넣기 시작한다. visualization mode가 simple이면 그냥 영역, full 이면 ground truth 도 출력하니 필요에 따라 출력모드를 바꾸면 된다.

주의사항: shell 로 실행하는 경우 X11_forwarding 기능이 필수다. 본 예제는 vscode remote shell로 했으니 참고해주기 바란다.

혹시 실행이 잘 안된다면, 아래 라이브러리도 같이 설치해본다. openCV도 같이 설치해 보고

```
%pip install Namespace
%pip install ArgumentParser
%pip install Trainer
%pip install DictConfig
%pip install Tensor
%pip install nn
%pip install Path
```

실행하면 이런 식으로 결과 그래프가 나오면서 이미지를 분류해서 넣을 것이다.

`Predicting DataLoader 0: 100%|████████████████████| 1/1 [00:00<00:00,  1.30it/s]`

## 문제

이렇게 로컬 환경의 추론을 잘 진행했으니 이대로 웹에 베포할 수 있겠지 만만세는 꿈이다.

우선 마지막 코드가 상당한 문제를 만드는데 이거 그냥 python bash다. 이대로는 절대는 아니지만, 상당 경우 api로 실행할 수 없다.

그렇기 때문에 저 위에서 실행한 파이썬 코드를 api 서버용으로 써야 하는데 그냥은 또 못쓴다.

따라서 inference를 수행하는 코드를 변형하여 서비스에 같이 넣어줘야 하므로 서비스 코드를 아예 새로 작성하게 된다.

## 라이브러리 고치기

일단 사용하는 코드는 lighting_inference.py 를 가져다가 사용한다.

```python
"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args: Namespace):
    """Run inference."""
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    # create the dataset
    dataset = InferenceDataset(
        args.input, image_size=tuple(config.dataset.image_size), transform=transform  # type: ignore
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    args = get_parser().parse_args()
    infer(args)

```

해당 코드를 이제 api에 맞춰서 수정하는 절차를 진행해야 한다. args 인자들을 다 fast api 내 서비스로 바꿀 것이다.

웹 서비스까지 같이 첨부하여 하나의 파이썬 코드로 작성되는 만큼 필요한 부분만 일단 먼저 소개하고자 한다.

```python
from argparse import Namespace
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

import torch
import io
import os 
from datetime import datetime
import shutil

from typing import List
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
templates = Jinja2Templates(directory="templates")

```

수많은 라이브러리와 패키지를 집어 넣고 app 으로 해당 서비스를 실행하도록 설정한다.

/output 부분에서 이미지를 호스팅하고 결과를 보여줘야 하기 때문에 /output을 설정해 준다.

FastAPI에 대해 제대로 된 지식이 있는게 아니다 보니 안타깝게도 해당 부분에 많은 삽질이 있었다.

추후 FastAPI를 사용할 일이 많다면 추가적인 학습이 어느정도 필요할 듯 하다.

인텔이 저작권을 가진 인퍼런스 스크립트를 수정하면 아래와 같은 스크립트가 나온다.

```python
@app.post("/predict", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
  
       # Save the uploaded image content to the 'input' directory
    file_path = os.path.join('./input', file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    args = Namespace(
        config='config.yaml',
        weights='model.ckpt',
        input='./input',
        output='./output',
        visualization_mode="full",
        show=False,
    )
    predictions = infer(args)

    image_paths = [pred['image_path'][0] for pred in predictions]

   # Construct the HTML content to display the predicted images
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predicted Images</title>
    </head>
    <body>
        <h1>Predicted Images</h1>
    """

    for image_path in image_paths:
        html_content += f'<img src="/output/{image_path}" alt="predicted image"><br>'

    html_content += """
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
```

먼저 post 를 통해 이미지를 업로드 받으면 사전 설정한 args를 해당 코드에 저장한 다음 infer 함수를 실행하도록 한다.

!python 으로 된 인자값을 이런 식으로 미리 지정하여 다 저장한 다음 출력하는 방식이다.

이미지를 넣고 결과가 나오면 간단한 이미지 결과를 내보내는 HTML 을 리턴하도록 한다.

```python
def infer(args: Namespace):
    """Run inference."""
    config = get_configurable_parameters(config_path=args.config)
    config.trainer.resume_from_checkpoint = str(args.weights)
    config.visualization.show_images = args.show
    config.visualization.mode = args.visualization_mode
    if args.output:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = args.output
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    # Create the dataset with the custom data loader function
    dataset = InferenceDataset(
        args.input,
        transform=transform,
        image_size=tuple(config.dataset.image_size),
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    predictions = trainer.predict(model=model, dataloaders=[dataloader])
  
    return predictions




```

Namespace 형식으로 된 args들을 분해하여 파이토치 라이트닝에 맞춰서 모델을 가져오고 config도 불러온다. 필요에 따라 이미지에 대한 변형을 가할 수도 있다. 이와 관련된 자세한 내용은 pytorch lighting inference 문서를 확인하는 편이 좋을 듯. 잘 모르고 쓰는 코드라 자세하게 설명할 수가 없다.

[Inference in Production — PyTorch Lightning 1.6.2 documentation](https://lightning.ai/docs/pytorch/1.6.2/common/production_inference.html)

재주가 좋은 분들은 그냥 torch.py 에 있는 인퍼런스 스크립트를 위처럼 수정해서 사용할 수도 있다.

torch 추론 스크립트는 segmentation 도 수행하는데, 본 프로젝트는 2개나 다 수행할 시간적, 정신적 여유가 부족하여 코드 칠줄 아시는 분이 직접 수행하는 편이 좋을 듯 하다.

정 안되면 gpt에서 적절하게 수정하도록 프롬프트를 넣어주면 위처럼 잘 짜 주지 않을까?

코드의 나머지 부분

```python

def dir_cleaning():
    # Create the base archive directories if they don't exist
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output/input', exist_ok=True)

    # Get the current date and time in YYYY-MM-DD format
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

     # Initialize a counter to keep track of uploads within the same second
    upload_counter = 0

    # Define the base directory names for input and output archives
    base_input_archive_dir = f"./archive/{current_datetime}"
    base_output_archive_dir = f"./archive/{current_datetime}"

    # Create the base archive directories if they don't exist
    os.makedirs(base_input_archive_dir, exist_ok=True)
    os.makedirs(base_output_archive_dir, exist_ok=True)

    # Create the subdirectory names using the upload counter
    input_archive_dir = f"{base_input_archive_dir}/input_{upload_counter}"
    output_archive_dir = f"{base_output_archive_dir}/output_{upload_counter}"

    # Increment the upload counter if the subdirectories already exist
    while os.path.exists(input_archive_dir) or os.path.exists(output_archive_dir):
        upload_counter += 1
        input_archive_dir = f"{base_input_archive_dir}/input_{upload_counter}"
        output_archive_dir = f"{base_output_archive_dir}/output_{upload_counter}"

    # Create the archive directories if they don't exist
    os.makedirs(input_archive_dir, exist_ok=True)
    os.makedirs(output_archive_dir, exist_ok=True)

    # Move the input image to the input archive directory
    shutil.move('./input', input_archive_dir)
    # Move the output image to the output archive directory (if needed)
    shutil.move('./output/input', output_archive_dir)
  

    # Create the base archive directories if they don't exist
    # 옮길 때 디렉터리 단위로 움직이므로 다시 생성해줘야 함 
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output/input', exist_ok=True)




@app.get("/predict", response_class=HTMLResponse)
async def upload_image_form():

    # cleaning pre trained result of images > move to archive 
    # 아카이브로 던지는 코드, 분 단위로 폴더 분류 
    dir_cleaning()

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/output/{image_path}")
async def get_result_image(image_path: str):
    # Since the {image_path} contains "input/", we need to adjust the file path
    # image_relative_path = image_path.replace("input/", "output/input/")
    image_file_path = os.path.join('/{$프로젝트의 루트 폴더, modify by system}/fastapi/api/output'+{image_path})

    return FileResponse(image_file_path)


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000)


```

나머지 부분은 머신러닝과는 그닥 관련은 없는데 기본적인 웹 호스팅과 파일 관리가 필요하여 다음과 같이 작성한다.

/predict로 접근할 때 이전에 넣은 input / output 데이터들을 전부 아카이브 항목으로 집어넣은 다음, 업로드를 진행하면 predict를 진행하고 그 결과를 리턴하는 작업을 수행한다. 파이썬으로 서버 만드는 사람들은 뭔 이렇게 서버를 만드냐 할 수 있는데, 이사람 파이썬 머신러닝 코드 볼 때마다 야매로 하나하나 익히는 식이라 파이썬의 데이터 구조나 동작 구조에 대해 거의 모른다 엌 ㅋㅋㅋ 그렇다고 자바나 스프링은 빠삭하게 아냐 하면 그것도 좀 자신이 없네

결국 이렇게 코드를 합쳐보면 링크에 있는 코드처럼 완성이 된다.

[SWbootProject_2023-7/2nd_cnn/fastapi/api/final.py at main · choi4624/SWbootProject_2023-7 (github.com)](https://github.com/choi4624/SWbootProject_2023-7/blob/main/2nd_cnn/fastapi/api/final.py)

## 추론 진행 


![](../assets/20230721_141341_2023-07-21_141331.png)

![](../assets/20230721_141213_2023-07-20_161031.png)

이와 같은 과정을 통해 테스트 환경에서 이미지를 업로드하면 추론 모델이 fault를 잡아내서 결과 이미지를 보여준다. 

cpu 대략 4개를 사용해서 5초 정도의 시간이 소요되므로, 대형 서비스가 아니라 소규모 테스트용으로는 gpu가 굳이 필요하진 않다. gpu 사용하는 법은 기재하지 않겠다.(torch의 경우 그냥 알아서 cuda장치 있으면 잡는 것으로 되어 있어서 로컬 환경의 torch 부분을 참고하면 될 듯) 참고로 gpu를 할당하는게 더 오래 걸릴 수도 있다. gpu 할당하고 task를 넣고 작동하는데 지연 시간이 발생할 수 있음. 물론 gpu 넣으면 귀한? cpu 자원을 덜 써도 된다는 점에서 나름의 장점이 있다.

## 공개 베포

날먹


글 작성자 시스템은 이미 엔진엑스를 통한 리버스 프록싱 시스템과 let's encrypt 서비스를 가지고 있으므로, 저거 로컬에서 실행하는 걸 그대로 프록시 연결해서 던져주면 된다. 


```nginx
server {
	listen [::]:443 ssl; # managed by Certbot
    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/ # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/ # managed by Certbot
    include /etc/ # managed by Certbot
    ssl_dhparam /etc/ # managed by Certbot


	root /var/www/html;

	# Add index.php to the list if you are using PHP
	index index.html index.htm index.nginx-debian.html;

	server_name rs.i4624.tk; ## 예시 URL 

	location / {
		# First attempt to serve request as file, then
		# as directory, then fall back to displaying a 404.
		proxy_pass http://192.168.0.140:8000/; 
                ## 내부 서비스 연결, fast api에서 https 요청을 받지 않아 http로 접속  
		proxy_buffers 16 4k;
    	proxy_buffer_size 2k;
    	proxy_set_header Host $host;
    	proxy_set_header X-Real-IP $remote_addr;
	}

}
```

이런식으로 http 리버스 프록시를 잡아주고 로컬에서 실행해주면 끝 

`nohup uvicorn final:app --host 0.0.0.0 --port 8000 --workers 1 &`

오래 켜놓을 서비스는 아니기 때문에 간단하게 실행 설정 해놓고 끝내면 된다. 

이렇게 설정을 해 놓으면 rs.i4624.tk/predict 로 접속하는 경우 로컬에서 봤던 이미지 업로드 기능을 웹 서비스 형태로 쓸 수 있게 된다.

여기서 좀 더 나가면 가상머신에 관련 서비스를 넣어 가볍게 베포하고 꾸미고 그러면 당신도 쓸모있는 이력서 거리를 할 수 있지 않을까? 

참고로 이 프로젝트를 좀 더 깍아서 엔드 유저도 쉽게 만들어준 프로젝트가 있으므로 참고해주기 바란다. 

[anomaly-detection-in-industry-manufacturing/anomalib_contribute at master · vnk8071/anomaly-detection-in-industry-manufacturing (github.com)](https://github.com/vnk8071/anomaly-detection-in-industry-manufacturing/tree/master/anomalib_contribute)

[openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. (github.com)](https://github.com/openvinotoolkit/anomalib)
