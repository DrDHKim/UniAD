"""
carla_trajectory_api.py — CARLA용 궤적 레이블 생성 API

NuScenesTraj의 CARLA 대체 구현.
NuScenes DB 없이 pkl info dict에서 직접 궤적 정보를 추출한다.

NuScenesTraj와의 핵심 차이점:
  - NuScenes: nusc.get('sample', token)['anns'] + PredictHelper로 future/past traj 계산
  - CarlaTraj: pkl info['fut_traj'] / info['fut_traj_valid_mask'] 직접 사용
               sdc vel / sdc future traj: 연속 프레임의 ego2global_translation 차분으로 계산
               past traj: pkl에 없으므로 zeros 반환 (모델이 zero mask로 invalid 처리)
"""

import numpy as np
from nuscenes.eval.common.utils import Quaternion
from nuscenes.prediction import convert_global_coords_to_local
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor


class CarlaTraj:
    """
    CARLA 데이터셋용 궤적 레이블 생성기.

    NuScenesTraj의 모든 public 메서드를 같은 시그니처로 제공하되,
    NuScenes DB 대신 pkl의 info dict에서 데이터를 읽는다.

    Attributes:
        data_infos (list[dict]): pkl에서 로드한 전체 info 리스트 (timestamp 오름차순 정렬 상태).
        token_to_idx (dict[str, int]): info['token'] → data_infos 인덱스 O(1) 조회.
        sdc_vel_info (dict[str, np.ndarray]):
            token → (2,) np.float32. 해당 프레임의 ego 속도 (LiDAR 좌표계, [vx, vy] m/s).
            연속 프레임의 ego2global_translation 차분으로 계산.
    """

    def __init__(self,
                 data_infos,
                 predict_steps,
                 planning_steps,
                 past_steps,
                 fut_steps,
                 with_velocity,
                 CLASSES,
                 box_mode_3d,
                 use_nonlinear_optimizer=False):
        """
        Args:
            data_infos      : pkl 'infos' 리스트 (timestamp 오름차순).
            predict_steps   : 미래 예측 스텝 수 (12 → 6초 @ 0.5s/step).
            planning_steps  : SDC 플래닝 스텝 수 (6 → 3초).
            past_steps      : 과거 스텝 수 (4 → 2초). CARLA pkl에는 없으므로 zeros 반환.
            fut_steps       : past_traj에 함께 담을 미래 스텝 수 (4).
            with_velocity   : True이면 bbox에 velocity 추가.
            CLASSES         : 클래스 이름 리스트 (nuScenes 10종 유지).
            box_mode_3d     : mmdet3d Box3DMode (LiDAR).
            use_nonlinear_optimizer: True이면 fut_traj 원점을 box center로 설정.
        """
        self.data_infos = data_infos
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.with_velocity = with_velocity
        self.CLASSES = CLASSES
        self.box_mode_3d = box_mode_3d
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        # token → data_infos 인덱스 조회 테이블
        # 프레임 간 이동(info['next'] / info['prev'])을 O(1)로 처리
        self.token_to_idx = {
            info['token']: i for i, info in enumerate(data_infos)
        }

        # SDC 속도 사전 계산 (NuScenesTraj.prepare_sdc_vel_info와 동일 역할)
        self.prepare_sdc_vel_info()

    def _compute_startup_first_move(self):
        """
        startup 씬(scene_0483~0637)별로 '최초 비정지 frame_idx'를 사전 계산.

        판정 기준: ego2global_translation 기반 프레임 간 이동거리 >= 0.1m이면 '이동 시작'.
        씬의 frame_idx=0부터 순회하여 처음으로 이동이 감지된 frame_idx를 반환.
        이동 없이 씬이 끝나면 마지막 frame_idx + 1 (전체 STOP).

        Returns:
            dict[int, int]: {scene_num: first_move_frame_idx}
        """
        import re
        # startup 씬의 info를 scene_num별로 그룹화 (frame_idx 순 정렬)
        startup_scenes = {}  # scene_num → list[(frame_idx, info_idx)]
        for i, info in enumerate(self.data_infos):
            cam_path = info['cams']['CAM_FRONT']['data_path']
            m = re.search(r'scene_(\d+)', cam_path)
            if not m:
                continue
            sn = int(m.group(1))
            if 483 <= sn <= 637:
                startup_scenes.setdefault(sn, []).append((info['frame_idx'], i))

        result = {}
        for sn, frames in startup_scenes.items():
            # frame_idx 오름차순 정렬
            frames.sort(key=lambda x: x[0])
            first_move = frames[-1][0] + 1  # 기본값: 전체 STOP (이동 없음)
            for j in range(1, len(frames)):
                prev_info = self.data_infos[frames[j-1][1]]
                curr_info = self.data_infos[frames[j][1]]
                # 프레임 간 이동거리 계산
                pos_prev = np.array(prev_info['ego2global_translation'][:2])
                pos_curr = np.array(curr_info['ego2global_translation'][:2])
                delta = np.linalg.norm(pos_curr - pos_prev)
                if delta >= 0.1:  # 0.1m 이상 이동 → 주행 시작
                    first_move = frames[j][0]  # 이 frame_idx부터 FORWARD
                    break
            result[sn] = first_move

        return result

    # ------------------------------------------------------------------
    # SDC 속도 계산
    # ------------------------------------------------------------------

    def prepare_sdc_vel_info(self):
        """
        전체 프레임에 대해 SDC(ego) 속도를 계산하여 self.sdc_vel_info에 저장.

        계산 방식:
            vel_global = (ego_xyz_next - ego_xyz_curr) / dt   [m/s, global 좌표계]
            vel_lidar  = vel_global @ inv(e2g_R).T @ inv(l2e_R).T  [LiDAR 좌표계 2D]

        NuScenesTraj.prepare_sdc_vel_info와 동일한 변환 순서.
        마지막 프레임(next == ''): 이전 프레임 속도를 복사 (NuScenes와 동일).

        CARLA 좌표:
            - ego2global_rotation : [w,x,y,z] 쿼터니언
            - lidar2ego_rotation  : [w,x,y,z] 쿼터니언 (CARLA에서는 [1,0,0,0] 항등)
        """
        self.sdc_vel_info = {}

        for info in self.data_infos:
            token = info['token']
            next_token = info['next']

            # next가 존재하고 같은 씬일 때만 속도 계산
            if (next_token != '' and
                    next_token in self.token_to_idx):
                next_info = self.data_infos[self.token_to_idx[next_token]]
                if next_info['scene_token'] == info['scene_token']:
                    # --- global 속도 계산 ---
                    dc = (np.array(next_info['ego2global_translation']) -
                          np.array(info['ego2global_translation']))  # (3,) [m]
                    dt = (next_info['timestamp'] - info['timestamp']) / 1e6  # [s]
                    dt = max(dt, 1e-6)  # 0 나눗셈 방지
                    vel_global = dc / dt  # (3,) [m/s]

                    # --- global → LiDAR 좌표계 변환 ---
                    # NuScenesTraj.prepare_sdc_vel_info와 동일:
                    #   vel_lidar = vel_global @ inv(e2g_R).T @ inv(l2e_R).T
                    e2g_r_mat = Quaternion(
                        info['ego2global_rotation']).rotation_matrix   # (3,3)
                    l2e_r_mat = Quaternion(
                        info['lidar2ego_rotation']).rotation_matrix     # (3,3)
                    vel_lidar = (vel_global
                                 @ np.linalg.inv(e2g_r_mat).T
                                 @ np.linalg.inv(l2e_r_mat).T)        # (3,)
                    # XY 성분만 저장 (3D bbox의 velocity channel과 동일 포맷)
                    self.sdc_vel_info[token] = vel_lidar[:2].astype(np.float32)
                    continue  # 다음 프레임으로

            # next 없음 또는 씬 경계: 임시로 zero 저장 (아래에서 이전 값으로 교체)
            self.sdc_vel_info[token] = np.array([0.0, 0.0], dtype=np.float32)

        # 씬 마지막 프레임: 직전 프레임 속도로 대체 (NuScenes 동일)
        for info in self.data_infos:
            token = info['token']
            next_token = info['next']
            # next가 없거나 씬이 바뀌면 이 프레임이 씬의 마지막
            is_last = (
                next_token == '' or
                next_token not in self.token_to_idx or
                self.data_infos[self.token_to_idx[next_token]]['scene_token']
                != info['scene_token']
            )
            if is_last:
                prev_token = info['prev']
                if prev_token != '' and prev_token in self.sdc_vel_info:
                    # 이전 프레임 속도로 대체 (NuScenes: second_last → last 복사)
                    self.sdc_vel_info[token] = self.sdc_vel_info[prev_token]

    # ------------------------------------------------------------------
    # Agent 궤적 레이블
    # ------------------------------------------------------------------

    def get_traj_label(self, info, mask):
        """
        현재 프레임의 agent 미래/과거 궤적 레이블 반환.

        NuScenesTraj.get_traj_label(sample_token, ann_tokens) 대체:
          - nuScenes: PredictHelper.get_future_for_agent() + convert_local_to_global
          - CARLA: pkl info['fut_traj'][mask] 직접 사용 (carla_to_nuscenes.py에서 미리 계산)

        Args:
            info (dict): 현재 프레임의 info dict.
            mask (np.ndarray[bool]): 유효한 agent 선택 마스크 (shape: N_total).

        Returns:
            fut_traj            (N, predict_steps, 2)  [m, ego/LiDAR 좌표계]
            fut_traj_valid_mask (N, predict_steps, 2)  0 or 1 (마지막 축 복제)
            past_traj           (N, past_steps+fut_steps, 2)  zeros (CARLA pkl 미저장)
            past_traj_valid_mask(N, past_steps+fut_steps, 2)  zeros (모두 invalid)

        Notes:
            fut_traj 좌표계: current ego frame (x=right, y=fwd) [m].
            lidar2ego_rotation = identity이므로 LiDAR frame과 동일.
            predict_steps=12, past_steps+fut_steps=8.
        """
        n_total = mask.sum()

        # --- 미래 궤적 ---
        # info['fut_traj']            : (N_total, 12, 2) float32, ego frame
        # info['fut_traj_valid_mask'] : (N_total, 12) int (0 or 1)
        fut_traj_raw = info['fut_traj'][mask]            # (N, 12, 2)
        valid_mask_1d = info['fut_traj_valid_mask'][mask]  # (N, 12)

        # predict_steps에 맞게 크기 조정 (보통 12 == predict_steps)
        T = fut_traj_raw.shape[1]
        if T < self.predict_steps:
            # 부족한 스텝은 zeros로 패딩
            pad_traj = np.zeros((n_total, self.predict_steps - T, 2),
                                dtype=np.float32)
            pad_mask = np.zeros((n_total, self.predict_steps - T),
                                dtype=np.float32)
            fut_traj_raw = np.concatenate([fut_traj_raw, pad_traj], axis=1)
            valid_mask_1d = np.concatenate([valid_mask_1d, pad_mask], axis=1)
        else:
            # 초과분 자르기
            fut_traj_raw = fut_traj_raw[:, :self.predict_steps, :]
            valid_mask_1d = valid_mask_1d[:, :self.predict_steps]

        # (N, predict_steps) → (N, predict_steps, 2) : xy 모두 같은 마스크
        fut_traj_valid_mask = np.stack(
            [valid_mask_1d, valid_mask_1d], axis=-1
        ).astype(np.float32)

        # --- 과거 궤적 ---
        # CARLA pkl에는 agent past traj 미저장 → zeros, 모두 invalid
        # 모델은 valid_mask=0인 스텝을 loss 계산에서 제외
        n_past = self.past_steps + self.fut_steps  # 4+4=8
        past_traj = np.zeros((n_total, n_past, 2), dtype=np.float32)
        past_traj_valid_mask = np.zeros((n_total, n_past, 2), dtype=np.float32)

        return (fut_traj_raw.astype(np.float32),
                fut_traj_valid_mask,
                past_traj,
                past_traj_valid_mask)

    # ------------------------------------------------------------------
    # SDC 의사 bbox (generate_sdc_info)
    # ------------------------------------------------------------------

    def generate_sdc_info(self, sdc_vel, as_lidar_instance3d_box=False):
        """
        SDC(ego 차량)를 나타내는 의사 3D bbox 생성.

        NuScenesTraj.generate_sdc_info와 완전히 동일.
        SDC는 항상 원점(0,0,0)에 위치하는 의사 박스로 표현.

        Args:
            sdc_vel (np.ndarray, (2,)): SDC 속도 [vx, vy] m/s (LiDAR frame).
            as_lidar_instance3d_box (bool):
                True  → LiDARInstance3DBoxes 객체 반환 (sdc_planning 계산용).
                False → DC 래핑된 (gt_sdc_bbox, gt_sdc_label) 반환 (학습용).

        Returns:
            (DC(gt_bboxes_3d), DC(gt_labels_3d)) or LiDARInstance3DBoxes.

        Notes:
            psudo_sdc_bbox: [x=0, y=0, z=0, l=4.08, w=1.73, h=1.56, yaw=π/2]
            SDC 치수: nuScenes 포럼 ego vehicle 치수 기준.
        """
        # SDC 의사 bbox: 원점, 차량 표준 크기, yaw=π/2 (mmdet3d 1.0.0rc6 포맷)
        psudo_sdc_bbox = np.array(
            [0.0, 0.0, 0.0, 4.08, 1.73, 1.56, 0.5 * np.pi],
            dtype=np.float32)
        if self.with_velocity:
            # velocity 채널 추가: [x,y,z,l,w,h,yaw,vx,vy]
            psudo_sdc_bbox = np.concatenate(
                [psudo_sdc_bbox, sdc_vel], axis=-1)

        gt_bboxes_3d = np.array([psudo_sdc_bbox], dtype=np.float32)  # (1, 7 or 9)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            gt_labels_3d.append(
                self.CLASSES.index(cat) if cat in self.CLASSES else -1)
        gt_labels_3d = np.array(gt_labels_3d)

        # nuScenes 박스 중심 (0.5,0.5,0.5) → KITTI 포맷 (0.5,0.5,0) 변환
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        if as_lidar_instance3d_box:
            return gt_bboxes_3d  # sdc_planning 계산 시 직접 조작

        # 학습용: DataContainer 래핑
        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)
        return gt_bboxes_3d, gt_labels_3d

    # ------------------------------------------------------------------
    # SDC 미래 궤적 (sdc_fut_traj)
    # ------------------------------------------------------------------

    def get_sdc_traj_label(self, sample_token):
        """
        SDC의 미래 궤적 레이블 생성.

        NuScenesTraj.get_sdc_traj_label 대체:
          - nuScenes: nusc.get('ego_pose', ...) 로 미래 프레임 위치 조회
          - CARLA: data_infos[token_to_idx[next_token]]['ego2global_translation'] 사용

        Args:
            sample_token (str): 현재 프레임 token.

        Returns:
            sdc_fut_traj_all       (1, predict_steps, 2)  current ego frame [m]
            sdc_fut_traj_valid_mask(1, predict_steps, 2)  0 or 1

        Notes:
            future global XY → convert_global_coords_to_local로 current ego frame 변환.
            nuscenes.prediction.convert_global_coords_to_local 사용 (동일 라이브러리).
        """
        info = self.data_infos[self.token_to_idx[sample_token]]
        # 현재 프레임의 ego global pose (변환 기준점)
        ego_start_trans = np.array(info['ego2global_translation'])   # (3,)
        ego_start_rot = info['ego2global_rotation']                   # [w,x,y,z]

        # 미래 프레임 global XY 수집
        sdc_fut_traj_list = []
        curr_info = info
        for _ in range(self.predict_steps):
            next_token = curr_info['next']
            if (next_token == '' or
                    next_token not in self.token_to_idx):
                break
            next_info = self.data_infos[self.token_to_idx[next_token]]
            # 씬 경계 초과 방지
            if next_info['scene_token'] != info['scene_token']:
                break
            # 미래 프레임의 global XY 위치 [m]
            sdc_fut_traj_list.append(
                np.array(next_info['ego2global_translation'][:2]))  # (2,)
            curr_info = next_info

        # 결과 배열 초기화 (유효 스텝이 없으면 zeros)
        sdc_fut_traj_all = np.zeros((1, self.predict_steps, 2), dtype=np.float32)
        sdc_fut_traj_valid_mask_all = np.zeros((1, self.predict_steps, 2),
                                               dtype=np.float32)
        n_valid = len(sdc_fut_traj_list)

        if n_valid > 0:
            sdc_fut_traj = np.stack(sdc_fut_traj_list, axis=0)  # (t, 2)

            # global → current ego local 좌표 변환
            # nuscenes.prediction.convert_global_coords_to_local:
            #   local = (global_xy - trans[:2]) @ rot_mat[:2,:2]
            sdc_fut_traj = convert_global_coords_to_local(
                coordinates=sdc_fut_traj,
                translation=ego_start_trans,
                rotation=ego_start_rot,
            )  # (t, 2) ego frame (x=right, y=fwd)

            sdc_fut_traj_all[0, :n_valid, :] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[0, :n_valid, :] = 1

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all

    # ------------------------------------------------------------------
    # SDC 플래닝 레이블 (sdc_planning)
    # ------------------------------------------------------------------

    def _get_l2g_transform(self, info):
        """
        info dict에서 LiDAR → global 변환 행렬 추출.

        NuScenesTraj.get_l2g_transform(sample) 대체:
          - nuScenes: nusc.get('sample_data', ...) + nusc.get('calibrated_sensor', ...) + nusc.get('ego_pose', ...)
          - CARLA: info dict에 lidar2ego_rotation/translation, ego2global_rotation/translation 직접 저장

        Returns:
            l2e_r_mat (3,3): LiDAR → ego 회전 행렬
            l2e_t     (3,):  LiDAR → ego 평행 이동 [m]
            e2g_r_mat (3,3): ego → global 회전 행렬
            e2g_t     (3,):  ego → global 평행 이동 [m]
        """
        l2e_r = info['lidar2ego_rotation']       # [w,x,y,z]
        l2e_t = np.array(info['lidar2ego_translation'])   # (3,) [m]
        e2g_r = info['ego2global_rotation']      # [w,x,y,z]
        e2g_t = np.array(info['ego2global_translation'])  # (3,) [m]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix  # (3,3)
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix  # (3,3)
        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def get_sdc_planning_label(self, sample_token):
        """
        SDC 플래닝 궤적 레이블 생성.

        NuScenesTraj.get_sdc_planning_label와 동일한 로직:
          1. 미래 각 스텝의 SDC bbox를 LiDAR frame에서 생성
          2. LiDAR → ego → world → initial ego → initial LiDAR 변환
          3. (x, y, yaw) 추출
          4. command: RIGHT(0)/LEFT(1)/FORWARD(2) 결정

        NuScenes와 다른 점: nusc.get() 대신 info dict + token_to_idx 사용.

        Args:
            sample_token (str): 현재 프레임 token.

        Returns:
            planning_all  (1, planning_steps, 3)  [(x,y,yaw) per step], initial LiDAR frame
            planning_mask (1, planning_steps, 2)  0 or 1
            command       int  0=RIGHT, 1=LEFT, 2=FORWARD
        """
        info = self.data_infos[self.token_to_idx[sample_token]]
        # 초기 프레임의 LiDAR→global 변환 (기준 좌표계)
        l2e_r_init, l2e_t_init, e2g_r_init, e2g_t_init = \
            self._get_l2g_transform(info)

        planning = []
        curr_info = info
        for _ in range(self.planning_steps):
            next_token = curr_info['next']
            if (next_token == '' or
                    next_token not in self.token_to_idx):
                break
            next_info = self.data_infos[self.token_to_idx[next_token]]
            if next_info['scene_token'] != info['scene_token']:
                break

            next_token_key = next_info['token']
            l2e_r_curr, l2e_t_curr, e2g_r_curr, e2g_t_curr = \
                self._get_l2g_transform(next_info)

            # 미래 프레임의 SDC pseudo bbox (현재 프레임 LiDAR 원점 기준)
            next_bbox3d = self.generate_sdc_info(
                self.sdc_vel_info[next_token_key],
                as_lidar_instance3d_box=True)   # LiDARInstance3DBoxes

            # LiDAR frame → ego frame (미래)
            next_bbox3d.rotate(l2e_r_curr.T)
            next_bbox3d.translate(l2e_t_curr)

            # ego frame (미래) → world frame
            next_bbox3d.rotate(e2g_r_curr.T)
            next_bbox3d.translate(e2g_t_curr)

            # world frame → initial ego frame (역변환: translate then rotate)
            next_bbox3d.translate(-e2g_t_init)
            next_bbox3d.rotate(np.linalg.inv(e2g_r_init).T)

            # initial ego frame → initial LiDAR frame (역변환)
            next_bbox3d.translate(-l2e_t_init)
            next_bbox3d.rotate(np.linalg.inv(l2e_r_init).T)

            planning.append(next_bbox3d)
            curr_info = next_info

        # 결과 배열 초기화
        planning_all = np.zeros((1, self.planning_steps, 3), dtype=np.float32)
        planning_mask_all = np.zeros((1, self.planning_steps, 2), dtype=np.float32)
        n_valid = len(planning)

        if n_valid > 0:
            # LiDARInstance3DBoxes → tensor → (x, y, yaw) 추출
            planning_tensors = [p.tensor.squeeze(0) for p in planning]
            planning_np = np.stack(planning_tensors, axis=0)  # (n_valid, 9)
            planning_np = planning_np[:, [0, 1, 6]]           # (n_valid, 3): x, y, yaw
            planning_all[0, :n_valid, :] = planning_np
            planning_mask_all[0, :n_valid, :] = 1

        # command 결정: 0=RIGHT, 1=LEFT, 2=FORWARD, 3=STOP
        # 세션 31: Stop 커맨드 도입 — GT 궤적의 전방 이동량(y)이 임계값 미만이면 Stop.
        # "정지 유지" 프레임을 Forward에서 분리하여, Forward가 항상 "전진" 의미만 갖게 함.
        # 온라인 추론에서는 Stop을 절대 전송하지 않으므로, 모델은 Forward일 때 항상 wp>0 예측.
        STOP_FWD_THRESH = 0.1  # y(전방) 이동량 임계값 [m], 0.1m 미만이면 Stop
        mask_1d = planning_mask_all[0].any(axis=1)  # (planning_steps,) 유효 스텝 마스크

        if mask_1d.sum() == 0:
            # 미래 프레임 없음 (씬 끝) → 기본 FORWARD
            command = 2  # FORWARD
        else:
            valid_steps = planning_all[0, mask_1d]  # (n_valid, 3): x, y, yaw
            # CARLA planning_all 좌표계 (방향별 실측 확인 완료):
            #   x = Forward (전방): North 주행 시 x≈22m, y≈0
            #   y = Left (측방):    GIF 검증으로 확인 (y>0이 좌회전)
            # nuScenes 원본(x=Right, y=Forward)과 축이 swap + y 부호 반전.
            #
            # LEFT/RIGHT 판정: y(=lateral) 기준
            #   y >= 2  → LEFT  (좌측 이동)
            #   y <= -2 → RIGHT (우측 이동)
            # STOP 판정: startup 씬(483~637)의 초기 정지 프레임만 적용
            #   → 아래 별도 블록에서 처리
            max_fwd = np.abs(valid_steps[:, 0]).max()  # x = 전방 최대 이동량 [m]

            # STOP: startup 씬(scene_0483~0637)의 **최초 연속 정지 구간**만.
            # 판정 방법: 씬별 "첫 번째 비정지 frame_idx"를 사전 계산(_startup_first_move)
            #   → frame_idx < 그 값이면 STOP, 아니면 FORWARD/LEFT/RIGHT.
            # 호출 순서 무관(shuffle-safe) — frame_idx 기반 정적 판정.
            import re
            _cam_path = info['cams']['CAM_FRONT']['data_path']
            _scene_m = re.search(r'scene_(\d+)', _cam_path)
            _scene_num = int(_scene_m.group(1)) if _scene_m else -1
            _is_startup = 483 <= _scene_num <= 637

            # 씬별 첫 번째 비정지 frame_idx 사전 계산 (1회만)
            if not hasattr(self, '_startup_first_move'):
                self._startup_first_move = self._compute_startup_first_move()

            _first_move = self._startup_first_move.get(_scene_num, 0)

            if _is_startup and info['frame_idx'] < _first_move:
                # startup 씬의 초기 정지 구간 (frame_idx < 첫 이동 프레임)
                command = 3  # STOP
            elif valid_steps[-1][1] >= 2:
                command = 1  # LEFT (y >= 2m → 좌측 이동)
            elif valid_steps[-1][1] <= -2:
                command = 0  # RIGHT (y <= -2m → 우측 이동)
            else:
                command = 2  # FORWARD

        return planning_all, planning_mask_all, command
