"""
carla_e2e_finetune.py — CARLA 데이터셋으로 UniAD Stage2 파인튜닝 config

base_e2e.py에서 다음만 변경:
  - dataset_type: NuScenesE2EDataset → CarlaE2EDataset
  - data_root: data/Custom_dataset/   (NAS 심볼릭 링크)
  - ann_file: carla_infos_{train,val}.pkl
  - lr: 2e-4 → 2e-5  (파인튜닝 보폭 축소)
  - total_epochs: 2
  - load_from: CARLA epoch_1 체크포인트 (normal 471씬 학습 완료, BUG-D 수정본)
  - train_pipeline: LoadMultiViewImageFromFilesInCeph에 local_cache_dir 추가
      → 씬 단위 로컬 SSD 캐시로 NFS random I/O blocking 방지 (2026-03-25)

모든 모델 구조 / 손실 설정은 base_e2e.py 그대로 상속.
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

# -------------------------------------------------------------------
# 로컬 SSD 캐시 설정 (NFS blocking 방지)
# -------------------------------------------------------------------
# NAS(data/Custom_dataset)에서 이미지를 읽을 때 씬 단위로 로컬 SSD에 먼저 복사한 뒤
# 로컬에서 읽는다. 한 번 복사한 씬은 max_cache_scenes 한도 안에서 재사용(LRU).
# 씬당 약 90MB.
#
# max_cache_scenes=9999: 사실상 무한 LRU → loading.py가 디스크 eviction(rmtree)을 하지 않음.
# 디스크 공간 관리는 prefetch_scenes.py 데몬이 전담 (mtime 기준 GB 제한 eviction).
# 이렇게 분리해야 데몬이 복사한 씬을 워커가 즉시 삭제하는 충돌을 방지할 수 있다.
local_cache_dir    = '/tmp/carla_img_cache'   # 로컬 SSD 임시 캐시 경로
max_cache_scenes   = 9999                      # 사실상 무한 — eviction은 prefetch_scenes.py 전담

# LoadMultiViewImageFromFilesInCeph에 캐시 파라미터를 추가한 파인튜닝 전용 pipeline.
# base_e2e.py의 train_pipeline에서 첫 번째 요소(이미지 로더)만 교체한다.
# 나머지 요소들(Augmentation, Annotation 로딩, 포맷 변환 등)은 base_e2e.py와 동일.
#
# NOTE: mmdet Python config 파싱 시 _base_의 변수를 자식 파일에서 직접 참조할 수 없으므로
#       base_e2e.py의 변수 값을 아래에 직접 복사한다. 값이 변경될 경우 양쪽 동기화 필요.
#   - occflow_grid_conf: base_e2e.py:64
#   - point_cloud_range: base_e2e.py:10
#   - class_names:       base_e2e.py:15
#   - img_norm_cfg:      base_e2e.py:13
_occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],   # BEV x 범위 및 해상도 (m)
    'ybound': [-50.0, 50.0, 0.5],   # BEV y 범위 및 해상도 (m)
    'zbound': [-10.0, 10.0, 20.0],  # BEV z 범위 (단일 bin)
}
_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # [x_min,y_min,z_min,x_max,y_max,z_max] (m)
_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
_img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],  # BGR mean (ImageNet 통계)
    std=[1.0, 1.0, 1.0],               # std=1 → mean subtraction만 적용
    to_rgb=False,                      # BGR 유지 (OpenCV 기본)
)

carla_train_pipeline = [
    # ── 이미지 로더: 씬 단위 로컬 캐시 활성화 ──────────────────────
    # local_cache_dir가 비어 있지 않으면 NAS 이미지를 씬 단위로 /tmp에 복사 후 읽는다.
    # 첫 접근 시 1회 복사, 이후 동일 씬은 로컬에서 읽어 NFS blocking이 발생하지 않는다.
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=dict(backend='disk'),
        img_root='',
        local_cache_dir=local_cache_dir,    # 로컬 캐시 ON (/tmp/carla_img_cache)
        max_cache_scenes=max_cache_scenes,  # 최대 20씬 유지 (LRU)
    ),
    # ── 아래는 base_e2e.py의 train_pipeline과 완전히 동일 ──────────
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,   # occ_flow GT 포함
        with_ins_inds_3d=True,   # instance index 포함
        ins_inds_add_1=True,     # ins_inds를 1부터 시작 (0=배경 예약)
    ),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=_occflow_grid_conf,  # BEV occupancy flow GT 생성 설정
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False,
    ),
    dict(type='ObjectRangeFilterTrack', point_cloud_range=_point_cloud_range),
    dict(type='ObjectNameFilterTrack',  classes=_class_names),
    dict(type='NormalizeMultiviewImage', **_img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=_class_names),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds', 'img',
            'timestamp', 'l2g_r_mat', 'l2g_t',
            'gt_fut_traj', 'gt_fut_traj_mask',
            'gt_past_traj', 'gt_past_traj_mask',
            'gt_sdc_bbox', 'gt_sdc_label',
            'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask',
            'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks',
            'gt_segmentation', 'gt_instance', 'gt_centerness',
            'gt_offset', 'gt_flow', 'gt_backward_flow',
            'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid',
            'gt_future_boxes', 'gt_future_labels',
            'sdc_planning', 'sdc_planning_mask', 'command',
        ],
    ),
]

# base_e2e.py의 data dict를 CarlaE2EDataset으로 교체
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=carla_train_pipeline,  # 캐시 적용 pipeline
        # BUG-M 수정: CarlaE2EDataset의 default queue_length가 base_e2e.py model(=3)과 달라
        # 명시하지 않으면 불일치 발생. base_e2e.py의 model queue_length=3과 맞춤.
        queue_length=3,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        queue_length=3,   # BUG-M: train과 일치
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        queue_length=3,   # BUG-M: train과 일치
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
load_from = '/home/donghyunkim/Desktop/01_Autonomous_Driving_Practice/E2E_Autonomous_Driving_Practice/models/uniad/projects/work_dirs/stage2_e2e/carla_e2e_finetune/epoch_1_20260324_bugD_fixed.pth'

checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

evaluation = dict(interval=5)   # 5 epoch마다 검증
