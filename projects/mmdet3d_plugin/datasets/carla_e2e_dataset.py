"""
carla_e2e_dataset.py — CARLA E2E 데이터셋 (nuScenes 포맷 호환)

NuScenesE2EDataset을 상속하되 NuScenes DB 의존성 3곳을 교체:
  1. __init__ : NuScenes(...) 인스턴스 생성 스킵, CarlaTraj + 더미 맵으로 대체
  2. get_ann_info : nusc.get('sample', token)['anns'] → info['fut_traj'] 직접 사용
  3. get_data_info: nusc.get('log', ...)['location'] → 더미 제로 맵으로 대체

나머지 로직(union2one, prepare_train_data, prepare_test_data, occ 관련)은 모두 부모 클래스 재사용.
"""

import copy
import pickle
import numpy as np
import torch
import mmcv

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .nuscenes_e2e_dataset import NuScenesE2EDataset
from .data_utils.carla_trajectory_api import CarlaTraj


@DATASETS.register_module()
class CarlaE2EDataset(NuScenesE2EDataset):
    """
    CARLA E2E 데이터셋.

    NuScenesE2EDataset의 모든 학습/평가 로직(BEV queue, occ flow, union2one 등)을
    그대로 재사용하면서 NuScenes DB 없이 CARLA pkl만으로 동작한다.

    교체 포인트:
      - NuScenes(...)    → 없음 (self.nusc 미사용)
      - NuScenesTraj     → CarlaTraj  (pkl info dict 직접 사용)
      - VectorizedLocalMap + nusc_maps → 더미 제로 텐서 (맵 손실 ≈0이지만 학습 진행)
    """

    def __init__(self,
                 queue_length=4,
                 bev_size=(200, 200),
                 patch_size=(102.4, 102.4),
                 canvas_size=(200, 200),
                 overlap_test=False,
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 use_nonlinear_optimizer=False,
                 lane_ann_file=None,
                 eval_mod=None,
                 is_debug=False,
                 len_debug=30,
                 enbale_temporal_aug=False,
                 occ_receptive_field=3,
                 occ_n_future=4,
                 occ_filter_invalid_sample=False,
                 occ_filter_by_valid_flag=False,
                 file_client_args=dict(backend='disk'),
                 *args,
                 **kwargs):
        """
        NuScenesE2EDataset.__init__와 동일한 서명.

        핵심 차이:
          - super() 호출을 NuScenesDataset(조부모) 수준으로 제한하여
            NuScenesE2EDataset의 NuScenes(...) 인스턴스화 블록을 건너뜀.
          - load_annotations 오버라이드로 CARLA pkl 로드 + version 호환.
          - CarlaTraj를 traj_api로 사용.
          - 맵 관련 객체(nusc_maps, vector_map) 미생성 → get_data_info에서 더미 처리.
        """
        # --- 부모 init 전에 설정 (load_annotations에서 사용) ---
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)
        self.is_debug = is_debug
        self.len_debug = len_debug

        # ★ NuScenesDataset.__init__ 호출 (NuScenesE2EDataset은 건너뜀)
        #   이 호출 안에서 Custom3DDataset.__init__ → self.load_annotations(ann_file)
        #   → CarlaE2EDataset.load_annotations 가 실행됨
        super(NuScenesE2EDataset, self).__init__(*args, **kwargs)

        # --- E2E 공통 속성 (NuScenesE2EDataset.__init__ 복사분) ---
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.scene_token = None
        self.lane_infos = (self.load_annotations(lane_ann_file)
                           if lane_ann_file else None)
        self.eval_mod = eval_mod
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        # --- 맵 관련 파라미터 (더미 텐서 생성에 사용) ---
        self.map_num_classes = 3    # divider / ped-crossing / boundary
        if canvas_size[0] == 50:
            self.thickness = 1
        elif canvas_size[0] == 200:
            self.thickness = 2
        else:
            raise ValueError(f'Unsupported canvas_size: {canvas_size}')
        self.angle_class = 36
        self.patch_size = patch_size
        self.canvas_size = canvas_size

        # ★ NuScenes 맵 객체 미생성 (get_data_info에서 더미 처리)
        # self.nusc      → 미생성
        # self.nusc_maps → 미생성
        # self.vector_map → 미생성

        # ★ CarlaTraj: NuScenesTraj 대체 (NuScenes DB 불필요)
        self.traj_api = CarlaTraj(
            data_infos=self.data_infos,
            predict_steps=predict_steps,
            planning_steps=planning_steps,
            past_steps=past_steps,
            fut_steps=fut_steps,
            with_velocity=self.with_velocity,
            CLASSES=self.CLASSES,
            box_mode_3d=self.box_mode_3d,
            use_nonlinear_optimizer=use_nonlinear_optimizer,
        )

        # --- Occ 관련 속성 ---
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False
        self.occ_receptive_field = occ_receptive_field
        self.occ_n_future = occ_n_future
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # hardcode: 평가에 영향

    # ------------------------------------------------------------------
    # load_annotations 오버라이드
    # ------------------------------------------------------------------

    def load_annotations(self, ann_file):
        """
        CARLA pkl 로드.

        NuScenesE2EDataset.load_annotations 대체:
          - file_client.get(ann_file.name) 대신 pickle.load(open()) 사용
            (ann_file은 config에서 str로 전달되므로 .name 속성 없음)
          - self.version = 'v1.0-trainval' 로 고정
            (NuScenesDataset의 eval_detection_configs가 이 값 기대)

        Args:
            ann_file (str): CARLA pkl 경로. e.g. 'data/infos/carla_infos_train.pkl'

        Returns:
            list[dict]: timestamp 오름차순 정렬된 info 리스트.
        """
        # Custom3DDataset은 open(path, 'rb') 파일 객체를 전달하므로 분기 처리:
        #   - 파일 객체(BufferedReader 등): .name 속성으로 경로 추출 후 다시 open
        #   - str / Path: 직접 open
        if hasattr(ann_file, 'read'):
            # 파일 객체: .name 속성이 실제 경로 문자열
            ann_file.close()
            with open(ann_file.name, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(str(ann_file), 'rb') as f:
                data = pickle.load(f)

        # timestamp 오름차순 정렬 후 load_interval 적용 (NuScenesDataset 동일)
        data_infos = list(
            sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]

        self.metadata = data['metadata']
        # 'v1.0-trainval' 로 고정: NuScenesDataset.evaluate()의 eval_set_map 키 호환
        self.version = 'v1.0-trainval'

        # --- velocity shortcut 대응: non-startup 연속 정지 구간 필터링 (세션 33) ---
        # 문제: FWD+speed≈0+wp≈0 프레임이 4,626개로 압도적 → 모델이 "FWD+정지→wp=0" 학습.
        # 해결: 비startup 씬에서 연속 5프레임+ speed<0.1 구간의 앞부분을 제거,
        #       마지막 2프레임만 유지 (정지→출발 전환 학습용).
        # startup 씬(483~637)은 STOP 라벨이 있으므로 필터링 대상에서 제외.
        import re
        SLOW_THRESH = 0.1       # speed 임계값 (m/s)
        MIN_RUN_TO_FILTER = 5   # 이 이상 연속 정지해야 필터링 적용
        KEEP_TAIL = 2           # 연속 정지 끝에서 유지할 프레임 수
        n_before = len(data_infos)

        # 씬별 그룹화
        scene_groups = {}       # scene_num → [(list_idx, frame_idx, speed)]
        for i, info in enumerate(data_infos):
            m = re.search(r'scene_(\d+)', info['cams']['CAM_FRONT']['data_path'])
            if not m:
                continue
            sn = int(m.group(1))
            if 483 <= sn <= 637:
                continue        # startup 씬 제외
            scene_groups.setdefault(sn, []).append(
                (i, info['frame_idx'], info['can_bus'][13]))

        # 제거 대상 인덱스 수집
        remove_indices = set()
        for sn, frames in scene_groups.items():
            frames.sort(key=lambda x: x[1])     # frame_idx 오름차순
            # 연속 정지 run 탐색
            run_start = None
            for j, (list_idx, fidx, speed) in enumerate(frames):
                if speed < SLOW_THRESH:
                    if run_start is None:
                        run_start = j
                else:
                    if run_start is not None:
                        run_len = j - run_start
                        if run_len >= MIN_RUN_TO_FILTER:
                            # run_start ~ (j - KEEP_TAIL - 1) 제거
                            for k in range(run_start, j - KEEP_TAIL):
                                remove_indices.add(frames[k][0])
                    run_start = None
            # 씬 끝까지 정지 중인 경우
            if run_start is not None:
                run_len = len(frames) - run_start
                if run_len >= MIN_RUN_TO_FILTER:
                    for k in range(run_start, len(frames) - KEEP_TAIL):
                        remove_indices.add(frames[k][0])

        if remove_indices:
            data_infos = [d for i, d in enumerate(data_infos) if i not in remove_indices]

        n_after = len(data_infos)
        if n_before != n_after:
            print(f'[CarlaE2EDataset] slow-frame 필터링: {n_before} → {n_after} '
                  f'(-{n_before - n_after}개, non-startup 연속 정지 구간)')

        return data_infos

    # ------------------------------------------------------------------
    # get_ann_info 오버라이드
    # ------------------------------------------------------------------

    def get_ann_info(self, index):
        """
        Agent 어노테이션 + 궤적 레이블 반환.

        NuScenesE2EDataset.get_ann_info 대체:
          ① nusc.get('sample', info['token'])['anns'] → info['fut_traj'] 직접 사용
          ② self.traj_api.get_traj_label(token, ann_tokens) →
             self.traj_api.get_traj_label(info, mask) (CARLA 시그니처)

        나머지 로직(gt_bboxes_3d 생성, with_velocity 처리, label 매핑)은
        NuScenesE2EDataset.get_ann_info와 동일.

        Args:
            index (int): data_infos 인덱스.

        Returns:
            dict: nuScenes E2E 학습에 필요한 어노테이션 키 포함.
        """
        info = self.data_infos[index]

        # --- 유효 agent 마스크 ---
        if self.use_valid_flag:
            mask = info['valid_flag']           # bool array (N_total,)
        else:
            mask = info['num_lidar_pts'] > 0    # LiDAR 포인트 있는 객체만

        gt_bboxes_3d = info['gt_boxes'][mask]       # (N, 7) or (N, 9) with vel
        gt_names_3d = info['gt_names'][mask]         # (N,) str
        gt_inds = info['gt_inds'][mask]              # (N,) int64 instance id

        # --- ★ 궤적 레이블: NuScenes ann_tokens 불필요 ---
        # CarlaTraj.get_traj_label(info, mask) 호출
        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = \
            self.traj_api.get_traj_label(info, mask)

        # --- SDC 의사 bbox ---
        sdc_vel = self.traj_api.sdc_vel_info[info['token']]  # (2,) LiDAR frame
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)

        # --- SDC 미래 궤적 ---
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = \
            self.traj_api.get_sdc_traj_label(info['token'])

        # --- SDC 플래닝 레이블 ---
        sdc_planning, sdc_planning_mask, command = \
            self.traj_api.get_sdc_planning_label(info['token'])

        # --- 클래스 레이블 정수 변환 ---
        gt_labels_3d = []
        for cat in gt_names_3d:
            gt_labels_3d.append(
                self.CLASSES.index(cat) if cat in self.CLASSES else -1)
        gt_labels_3d = np.array(gt_labels_3d)

        # --- velocity 추가 ---
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]   # (N, 2) [vx, vy] m/s
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate(
                [gt_bboxes_3d, gt_velocity], axis=-1)  # (N, 9)

        # nuScenes 박스 중심 (0.5,0.5,0.5) → KITTI 포맷 (0.5,0.5,0) 변환
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,   # 0=RIGHT, 1=LEFT, 2=FORWARD, 3=STOP
        )

        # 정합성 검증 (NuScenesE2EDataset과 동일)
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

    # ------------------------------------------------------------------
    # get_data_info 오버라이드
    # ------------------------------------------------------------------

    def get_data_info(self, index):
        """
        단일 프레임의 입력 데이터 dict 구성.

        NuScenesE2EDataset.get_data_info 대체:
          ★ 맵 생성 블록 교체:
            - nusc.get('log', ...)['location'] → 필요 없음
            - VectorizedLocalMap.gen_vectorized_samples() → 더미 제로 텐서
            - obtain_map_info(self.nusc, ...) → 더미 제로 텐서
          나머지 (camera extrinsics/intrinsics, l2g transform, occ 관련)는
          NuScenesE2EDataset.get_data_info와 동일.

        더미 맵 처리:
            instance_masks, map_mask 모두 zeros
            → gt_lane_labels / gt_lane_bboxes / gt_lane_masks 모두 빈 텐서
            → 맵 세그멘테이션 손실 ≈ 0 (학습 진행에 지장 없음)

        Args:
            index (int): data_infos 인덱스.

        Returns:
            dict: pipeline에 전달할 입력 데이터 dict.
        """
        info = self.data_infos[index]

        # --- ★ 더미 맵 생성 (nusc + VectorizedLocalMap 대체) ---
        canvas_h, canvas_w = self.canvas_size   # (200, 200)
        # gt_lane_labels / gt_lane_bboxes / gt_lane_masks: 빈 텐서
        # NuScenesE2EDataset에서 이 3개를 input_dict에 담아 pipeline으로 전달
        gt_labels = torch.zeros(0, dtype=torch.long)           # (0,)
        gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)   # (0, 4)
        gt_masks = torch.zeros(
            (0, canvas_h, canvas_w), dtype=torch.uint8)        # (0, H, W)

        # --- 기본 입력 dict (NuScenesE2EDataset과 동일 키) ---
        lane_info = (self.lane_infos[index]
                     if self.lane_infos else None)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,   # μs → s
            map_filename=(lane_info['maps']['map_mask']
                          if lane_info else None),
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
        )

        # --- LiDAR → global 변환 행렬 (l2g_r_mat, l2g_t) ---
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix   # (3,3)
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix   # (3,3)

        # l2g_r_mat = L→E 역행렬 @ E→G 역행렬 = (l2e^T) @ (e2g^T)
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T   # (3,3)
        # l2g_t = LiDAR 원점을 global로 변환한 위치
        l2g_t = (np.array(l2e_t) @ e2g_r_mat.T
                 + np.array(e2g_t))               # (3,)

        input_dict.update(dict(
            l2g_r_mat=l2g_r_mat.astype(np.float32),
            l2g_t=l2g_t.astype(np.float32),
        ))

        # --- 카메라 extrinsics / intrinsics ---
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                # LiDAR → 카메라 변환
                # sensor2lidar_rotation: 카메라→LiDAR 회전 → 역행렬로 LiDAR→카메라
                lidar2cam_r = np.linalg.inv(
                    cam_info['sensor2lidar_rotation'])   # (3,3)
                lidar2cam_t = (np.array(cam_info['sensor2lidar_translation'])
                               @ lidar2cam_r.T)          # (3,)
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                # camera intrinsic → 4×4 패딩
                intrinsic = cam_info['cam_intrinsic']    # (3,3)
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0],
                        :intrinsic.shape[1]] = intrinsic

                lidar2img_rt = viewpad @ lidar2cam_rt.T  # (4,4)
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        # --- 어노테이션 (train/test 공통으로 항상 로드) ---
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # sdc_planning 관련 키 최상위로 복사 (pipeline의 CustomCollect3D에서 수집)
        if 'sdc_planning' in annos:
            input_dict['sdc_planning'] = annos['sdc_planning']
            input_dict['sdc_planning_mask'] = annos['sdc_planning_mask']
            input_dict['command'] = annos['command']

        # --- can_bus 업데이트 ---
        # nuScenes 형식: can_bus[:3] = ego 위치, can_bus[3:7] = 쿼터니언
        # NOTE: NuScenesE2EDataset.get_data_info 동일 처리
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation.elements     # [w,x,y,z]
        # patch_angle: BEVFormer can_bus yaw 값 ([-2π, 2π] → [0°, 360°])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi   # [rad]
        can_bus[-1] = patch_angle                  # [deg]

        # --- Occ 관련 데이터 (부모 메서드 재사용) ---
        (all_frames, has_invalid_frame,
         occ_transforms, occ_future_ann_infos) = self.get_occ_data_infos(index)

        input_dict['occ_has_invalid_frame'] = has_invalid_frame
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0
        input_dict.update(occ_transforms)
        input_dict['occ_future_ann_infos'] = occ_future_ann_infos

        return input_dict
