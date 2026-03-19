# STAC COCO 10% Quick (PyTorch Port Spec)

이 폴더는 STAC 원본(TensorFlow/tensorpack) 설정을 PyTorch로 옮길 때
값이 틀어지지 않도록 `COCO 10% + quick schedule`의 세부 값을 고정해둔 사양입니다.

파일:
- `stac_coco10_quick.py`: 원본 스크립트/논문 Appendix A 기준 파라미터와 파생 스케줄 계산 로직
- `stac_coco10_quick_stage1.py`: stage1(라벨 데이터 supervised 학습) 설정 분리본
- `stac_coco10_quick_predict.py`: pseudo label 생성(predict) 설정 분리본
- `stac_coco10_quick_stage2.py`: stage2(STAC 학습) 설정 분리본
- `pytorch_stac_common.py`: PyTorch 공통 유틸(dataset, schedule, model builder)
- `pytorch_stage1_train.py`: 실제 stage1 학습 스크립트
- `pytorch_predict_pseudo.py`: 실제 pseudo label 생성 스크립트
- `pytorch_stage2_train.py`: 실제 stage2 학습 스크립트
- `d2_stac_common.py`: Detectron2 공통 유틸
- `d2_stage1_train.py`: Detectron2 stage1
- `d2_predict_pseudo.py`: Detectron2 pseudo 생성
- `d2_stage2_train.py`: Detectron2 stage2
- `mmdet/config_faster_rcnn_coco_1x_stand.py`: MMDetection 표준 1x 설정
- `mmdet/train_stage1.py`: MMDetection stage1
- `mmdet/generate_pseudo.py`: MMDetection pseudo 생성
- `mmdet/train_stage2.py`: MMDetection stage2
- `mmdet/make_stage2_merged_json.py`: stage2 병합 annotation 생성
- `mmdet/estimate_runtime.py`: 러닝타임 추정 유틸

## 포함한 원본 기준
- 논문(STAC.pdf) Appendix A.1 quick schedule:
  - LR decay: `0.01 (<=120k), 0.001 (<=160k), 0.0001 (<=180k)`
  - `short edge = [500, 800]`
  - `FRCNN batch per image = 64`
- 저장소 스크립트(`detection/scripts/coco/*.sh`):
  - stage1/stage2/eval(pseudo) 실행 시 넘기는 오버라이드 값
  - stage2 핵심값: `TRAIN.CONFIDENCE=0.9`, `TRAIN.WU=2`, unlabeled strong aug

## 사용
```bash
python3 torch/stac_coco10_quick.py
python3 torch/stac_coco10_quick_stage1.py
python3 torch/stac_coco10_quick_predict.py
python3 torch/stac_coco10_quick_stage2.py
```

## PyTorch 실행 (실제 학습 파이프라인)
아래 3단계를 원본 STAC 흐름처럼 순서대로 실행합니다.

```bash
# 1) stage1 supervised
python3 torch/pytorch_stage1_train.py \
  --coco-dir ${COCODIR} \
  --output-dir torch/outputs/coco10_quick/stage1

# 2) pseudo label 생성 (stage1 마지막 체크포인트 사용)
python3 torch/pytorch_predict_pseudo.py \
  --coco-dir ${COCODIR} \
  --checkpoint torch/outputs/coco10_quick/stage1/model-180000.pth \
  --output torch/outputs/coco10_quick/pseudo_data.json

# 3) stage2 STAC
python3 torch/pytorch_stage2_train.py \
  --coco-dir ${COCODIR} \
  --pseudo-json torch/outputs/coco10_quick/pseudo_data.json \
  --load-stage1 torch/outputs/coco10_quick/stage1/model-180000.pth \
  --output-dir torch/outputs/coco10_quick/stage2
```

원본 스크립트 스타일로도 실행 가능합니다:
```bash
bash torch/scripts/coco/train_stg1.sh
bash torch/scripts/coco/eval_stg1.sh
bash torch/scripts/coco/train_stg2.sh
# or
bash torch/scripts/coco/train_stac.sh
```

필수 패키지:
```bash
pip install torch torchvision pycocotools pillow
```

## Detectron2 기반 실행 (RTX 3070 8GB 권장 기본값)
기본값은 `IMS_PER_BATCH=1`, `NUM_WORKERS=2`, `AMP=ON`으로 맞춰져 있습니다.

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# stage1
bash torch/scripts/coco/d2_train_stg1.sh

# pseudo 생성
bash torch/scripts/coco/d2_eval_stg1.sh

# stage2
bash torch/scripts/coco/d2_train_stg2.sh

# or one-shot
bash torch/scripts/coco/d2_train_stac.sh
```

메모:
- Detectron2 구현에서는 STAC의 `WU=2`를 loss 항 직접 가중 대신 pseudo annotation 오버샘플링(반복)으로 근사합니다.
- `--max-iter-scale`로 학습 길이를 줄여 빠르게 실험할 수 있습니다. 예: `--max-iter-scale 0.25`
- STAC 수동 세팅 반영:
  - `ROI_HEADS.BATCH_SIZE_PER_IMAGE=64` (원본 `FRCNN.BATCH_PER_IM=64`)
  - `MIN_SIZE_TRAIN=(500, 800)` + `MIN_SIZE_TRAIN_SAMPLING=choice`
  - Detectron2 기본 pretrained는 `imagenet`(원본 의도와 동일)로 설정
- step 수는 batch=8 기준으로 환산되어 `scale=(8/IMS_PER_BATCH)*max_iter_scale` 적용
    (예: `IMS_PER_BATCH=1`이면 `MAX_ITER=180000*8`)

## MMDetection 기반 실행 (표준 1x 스케줄)
요청하신 대로 STAC quick가 아니라 MMDetection 표준 1x(12 epoch, 8/11 decay) 기반입니다.

### COCO 10% 재현 가이드 (새 conda 환경, 1~8 단계)

1. conda 환경 생성
```bash
conda create -n stac-mmdet python=3.10 -y
conda activate stac-mmdet
```

2. PyTorch 설치 (CUDA 버전에 맞게 선택)
```bash
# 예시: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. MMDetection 의존성 설치
```bash
pip install -U openmim
mim install mmengine mmcv mmdet
pip install opencv-python pycocotools
```

4. COCO 2017 다운로드 및 압축 해제 (`$COCODIR`)
```bash
export COCODIR=/path/to/coco
mkdir -p "${COCODIR}"
cd "${COCODIR}"

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

5. STAC용 COCO 10% split 생성 (원본 방식)
```bash
cd /home/deep/DEEP/hy/ssl_detection
python3 prepare_datasets/prepare_coco_data.py --seed 1 --percent 10 --version 2017
```

6. split 생성 결과 확인
```bash
ls -lh "${COCODIR}/annotations/semi_supervised"/instances_train2017.1@10*.json
```
- `seed=1`, `percent=10` 기준 예시:
  - labeled: `11285 images`
  - unlabeled: `107002 images`

7. MMDetection stage별 실행
```bash
# stage1 (supervised)
bash torch/scripts/coco/mmdet_train_stg1.sh

# pseudo label 생성
bash torch/scripts/coco/mmdet_eval_stg1.sh
# pseudo 시각화 저장(선택)
VIS_DIR=torch/outputs_mmdet/coco10/PSEUDO_DATA/vis bash torch/scripts/coco/mmdet_eval_stg1.sh

# stage2 (labeled + pseudo)
bash torch/scripts/coco/mmdet_train_stg2.sh
```

8. one-shot 실행 (권장)
```bash
bash torch/scripts/coco/mmdet_train_stac.sh
```

참고:
- 기본 출력 경로:
  - stage1: `torch/outputs_mmdet/coco10/stage1_1x`
  - pseudo: `torch/outputs_mmdet/coco10/PSEUDO_DATA`
  - stage2: `torch/outputs_mmdet/coco10/stage2_1x`
- 짧은 테스트:
```bash
MAX_EPOCHS=1 bash torch/scripts/coco/mmdet_train_stac.sh
```

런타임 추정 예시:
```bash
# COCO 10% labeled (약 11.8k) 기준 stage1, 12 epoch, bs=1
python3 torch/mmdet/estimate_runtime.py --num-images 11800 --epochs 12 --batch-size 1

# COCO train 전체에 가까운 stage2 (약 118k) 기준, 12 epoch, bs=1
python3 torch/mmdet/estimate_runtime.py --num-images 118000 --epochs 12 --batch-size 1
```

출력 JSON에 다음이 들어있습니다:
- `stage1_overrides`
- `stage2_overrides`
- `pseudo_overrides`
- `derived_schedule_default_gpus` (원본 코드와 같은 GPU 스케일링 적용 결과)

## MMDetection 실험 요약 (stage1)
아래 표는 현재까지 진행한 stage1 실험의 주요 하이퍼파라미터와 **best mAP (val 기준)**을 정리한 것입니다.

| run | work_dir | load_from | norm | batch | accum | lr | warmup_end | milestones | max_epochs | early_stop(min_delta/patience) | best mAP | mAP50 | mAP75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260309_230427 | stage1_1x_retrain_20260309_230422 | pretrained | BN | 1 | - | 0.00125 | 500 | [8, 11] | 12 | - | 0.213 | 0.398 | 0.205 |
| 20260313_155446 | stage1_scratch | scratch | BN | 4 | - | 0.00125 | 500 | [8, 11] | 12 | - | 0.004 | 0.011 | 0.002 |
| 20260314_235901 | stage1_scratch_b2_m25 | scratch | BN | 2 | - | 0.00125 | 500 | [16, 22] | 25 | 0.002/5 | 0.021 | 0.051 | 0.013 |
| 20260315_223503 | stage1_scratch_b4_m50 | scratch | BN | 2 | 2 | 0.00125 | 500 | [33, 45] | 50 | 0.002/5 | 0.019 | 0.048 | 0.011 |
| 20260316_184847 | stage1_scratch_b2_acc2_m30_fast_lr25 | scratch | GN | 2 | 2 | 0.00250 | 200 | [18, 26] | 30 | 0.003/3 | 0.044 | 0.105 | 0.030 |
| 20260317_105826 | stage1_scratch_b2_acc2_m30_gn_lr125_wu500_fullrun | scratch | GN | 2 | 2 | 0.00250 | 200 | [18, 26] | 30 | - | 0.046 | 0.109 | 0.030 |
| 20260318_143933 | stage1_scratch_b2_acc2_m30_gn_lr125_wu500_es002_p4_m1422 | scratch | GN | 2 | 2 | 0.00250 | 200 | [16, 24] | 30 | 0.002/6 | 0.036 | 0.087 | 0.024 |

## 주의점 (논문 vs 저장소)
- 논문 quick는 `max_size=1024`로 설명됩니다.
- 원본 저장소의 COCO 스크립트는 `PREPROC.MAX_SIZE`를 따로 override하지 않아 기본값 `1333`이 유지됩니다.
- 파일에는 두 값을 모두 기록(`max_size_paper`, `max_size_repo`)해 두었습니다.
