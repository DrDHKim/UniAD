"""
carla_e2e_finetune.py — CARLA 데이터셋으로 UniAD Stage2 파인튜닝 config

base_e2e.py에서 다음만 변경:
  - dataset_type: NuScenesE2EDataset → CarlaE2EDataset
  - data_root: data/nuscenes/ → data/Custom_dataset/
  - ann_file: carla_infos_{train,val}.pkl
  - lr: 2e-4 → 2e-5  (파인튜닝 보폭 축소)
  - total_epochs: 20 → 5
  - load_from: UniAD nuScenes 사전학습 체크포인트

모든 모델 구조 / 파이프라인 / 손실 설정은 base_e2e.py 그대로 상속.
"""

_base_ = ['./base_e2e.py']

# -------------------------------------------------------------------
# 데이터셋 설정
# -------------------------------------------------------------------
dataset_type = 'CarlaE2EDataset'
data_root    = 'data/Custom_dataset/'   # CARLA 이미지/LiDAR 루트
info_root    = 'data/infos/'

ann_file_train = info_root + 'carla_infos_train.pkl'
ann_file_val   = info_root + 'carla_infos_val.pkl'

# base_e2e.py의 data dict를 CarlaE2EDataset으로 교체
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
    ),
)

# -------------------------------------------------------------------
# 파인튜닝 옵티마이저 (nuScenes 대비 LR 1/10)
# -------------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=2e-5,                     # nuScenes 2e-4의 1/10 (파인튜닝 표준)
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,       # 짧은 warmup (전체 이터가 적으므로)
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# -------------------------------------------------------------------
# 체크포인트 / 로그
# -------------------------------------------------------------------
# UniAD nuScenes 사전학습 체크포인트에서 시작 (Stage2 최종 체크포인트 경로)
# 실행 전 경로 확인 후 수정:
#   python models/uniad/tools/train.py \
#       models/uniad/projects/configs/stage2_e2e/carla_e2e_finetune.py \
#       --load-from <checkpoint_path>
load_from = '/home/donghyunkim/Desktop/01_Autonomous_Driving_Practice/E2E_Autonomous_Driving_Practice/models/uniad/ckpts/uniad_base_e2e.pth'

checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

evaluation = dict(interval=5)   # 5 epoch마다 검증
