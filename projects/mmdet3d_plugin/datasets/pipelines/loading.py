import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet3d.datasets.pipelines import LoadAnnotations3D
import os
import re
import shutil
import threading
from collections import OrderedDict

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        local_cache_dir (str): NAS→로컬 캐시 디렉토리 경로.
            빈 문자열('')이면 캐시 비활성화(기본값).
            지정하면 NAS 이미지를 씬 단위로 로컬 SSD에 복사 후 읽어서
            NFS random I/O blocking을 방지한다.
        max_cache_scenes (int): 로컬에 유지할 최대 씬 수 (LRU 방식).
            씬당 약 96MB이므로 기본값 20 = 약 2GB.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 img_root='',
                 local_cache_dir='',
                 max_cache_scenes=20):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.img_root = img_root

        # ── 로컬 캐시 설정 ──────────────────────────────────────────────
        # local_cache_dir가 지정된 경우에만 캐시를 활성화한다.
        # 빈 문자열이면 기존 동작(NAS 직접 읽기)을 유지한다.
        self.local_cache_dir = local_cache_dir
        self.max_cache_scenes = max_cache_scenes

        if self.local_cache_dir:
            # 캐시 디렉토리 생성 (이미 있으면 무시)
            os.makedirs(self.local_cache_dir, exist_ok=True)
            # LRU 캐시: {scene_id: 로컬_씬_디렉토리_경로} — OrderedDict로 접근 순서 추적
            # 멀티프로세스 워커에서 각자 독립적으로 관리하므로 Lock만 사용
            self._scene_cache = OrderedDict()
            self._cache_lock = threading.Lock()

    # 완전한 씬으로 인정하기 위해 존재해야 하는 최소 카메라 디렉토리 목록
    # 이 중 하나라도 jpg가 없으면 불완전(복사 중단) 씬으로 간주 → 재복사
    _REQUIRED_CAMS = (
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
    )

    def _is_scene_complete(self, local_scene_dir):
        """로컬 씬 디렉토리가 완전히 복사됐는지 확인한다.

        판단 기준:
          1. _REQUIRED_CAMS의 모든 카메라 디렉토리가 존재해야 함
          2. 각 카메라의 jpg 파일 수가 0이 아니어야 함
          3. 모든 카메라의 jpg 파일 수가 동일해야 함
             → 복사 중단 시 나중 카메라일수록 파일이 적어서 수가 달라지는 패턴을 감지

        예) CAM_FRONT=40, CAM_BACK_RIGHT=33 → set({40,33}) → len>1 → False → 재복사

        로컬 SSD 조회이므로 성능 영향은 미미하다.

        Args:
            local_scene_dir (str): 로컬 씬 루트 경로 (예: /tmp/carla_img_cache/scene_0065)

        Returns:
            bool: True이면 완전한 씬, False이면 불완전(재복사 필요)
        """
        samples_dir = os.path.join(local_scene_dir, 'samples')
        if not os.path.isdir(samples_dir):
            return False  # samples/ 자체가 없음

        counts = []  # 카메라별 jpg 파일 수 — 모두 같아야 완전

        for cam in self._REQUIRED_CAMS:
            cam_dir = os.path.join(samples_dir, cam)
            if not os.path.isdir(cam_dir):
                return False  # 카메라 디렉토리 자체가 없음

            # 해당 카메라 디렉토리의 jpg 파일 수
            count = sum(1 for f in os.listdir(cam_dir) if f.endswith('.jpg'))
            if count == 0:
                return False  # 파일이 아예 없음 (복사 시작 전 중단)
            counts.append(count)

        # 모든 카메라의 파일 수가 같아야 완전 복사된 씬
        # 복사 중단 시 순서상 나중 카메라일수록 파일이 적어 수가 달라짐
        return len(set(counts)) == 1

    def _get_scene_id(self, img_path):
        """img_path에서 'scene_XXXX' 식별자를 추출한다.

        Args:
            img_path (str): 이미지 절대 경로
                예) /mnt/nas/.../CARLA_scenes/scene_0065/samples/CAM_FRONT/000036.jpg

        Returns:
            str | None: 'scene_0065' 형태의 씬 ID, 매칭 실패 시 None
        """
        m = re.search(r'(scene_\d+)', img_path)
        return m.group(1) if m else None

    def _get_scene_nas_dir(self, img_path, scene_id):
        """img_path로부터 NAS 상의 씬 루트 디렉토리 경로를 계산한다.

        예) /mnt/nas/.../CARLA_scenes/scene_0065/samples/CAM_FRONT/000036.jpg
            → /mnt/nas/.../CARLA_scenes/scene_0065

        Args:
            img_path (str): 이미지 절대 경로
            scene_id (str): 'scene_0065' 형태의 씬 ID

        Returns:
            str: NAS 씬 루트 디렉토리 경로
        """
        # scene_id 바로 뒤 '/' 까지만 자른다
        idx = img_path.index(scene_id)
        return img_path[:idx + len(scene_id)]

    def _ensure_scene_cached(self, img_path):
        """NAS img_path가 속한 씬을 로컬 캐시에 복사하고 로컬 경로를 반환한다.

        동작 순서:
          1. img_path에서 scene_id 추출
          2. LRU 캐시에 scene_id가 있고 디렉토리도 실제로 존재하면 → 캐시 히트
          3. LRU에 있으나 디렉토리가 없으면 → race condition (다른 워커 rmtree)
             → LRU에서 제거 후 재복사
          4. LRU에 없으면 → NAS → 로컬 복사 (tmp → atomic rename)
             → 두 워커가 동시에 복사하려 해도 rename 충돌로 안전하게 처리
          5. LRU 크기 초과 시 → 가장 오래된 씬 로컬 삭제
          6. 로컬 경로로 변환된 img_path 반환

        Args:
            img_path (str): NAS 상의 원본 이미지 절대 경로

        Returns:
            str: 로컬 SSD 상의 동일 이미지 경로 (캐시 히트/미스 모두)
        """
        scene_id = self._get_scene_id(img_path)
        if scene_id is None:
            # scene_id 추출 실패 → 원본 경로 그대로 사용 (NFS 직접 읽기)
            return img_path

        with self._cache_lock:
            local_scene_dir = os.path.join(self.local_cache_dir, scene_id)
            # 디렉토리 존재 여부만이 아닌 완전성까지 확인한다.
            # os.path.isdir만으로는 복사 중단된 불완전 씬을 감지하지 못해
            # "파일 없음" crash가 발생할 수 있다.
            scene_complete = self._is_scene_complete(local_scene_dir)

            if scene_id in self._scene_cache and scene_complete:
                # ── 완전한 캐시 히트: LRU 순서 갱신 ────────────────────
                self._scene_cache.move_to_end(scene_id)

            else:
                # ── 캐시 미스, 디렉토리 삭제됨, 또는 불완전 씬 감지 ────

                if scene_id in self._scene_cache and not scene_complete:
                    # LRU에 등록됐지만 씬이 불완전한 경우
                    # 원인: rmtree race condition 또는 복사 중단 잔재
                    # → LRU dict에서 제거 후 아래에서 재복사
                    del self._scene_cache[scene_id]

                if not scene_complete:
                    # 불완전한 씬 디렉토리가 남아있으면 먼저 삭제
                    if os.path.isdir(local_scene_dir):
                        shutil.rmtree(local_scene_dir)
                    # ── NAS → 로컬 복사 (pid별 tmp → atomic rename) ─────
                    # threading.Lock은 같은 프로세스 내 스레드만 보호한다.
                    # DataLoader workers는 별개 프로세스이므로 Lock이 공유되지 않아
                    # 여러 워커가 동시에 같은 씬을 복사하려 시도할 수 있다.
                    #
                    # 해결: tmp 디렉토리 이름에 pid를 포함 → 워커마다 고유한 tmp 경로
                    #   worker 0: scene_0619.tmp.12345
                    #   worker 1: scene_0619.tmp.12346
                    # → 충돌 없이 각자 복사 후 atomic rename 시도
                    # → 한 워커만 rename 성공, 나머지는 OSError → 자신의 tmp 정리
                    nas_scene_dir = self._get_scene_nas_dir(img_path, scene_id)
                    tmp_dir = f'{local_scene_dir}.tmp.{os.getpid()}'

                    try:
                        if os.path.exists(tmp_dir):
                            # 이전 실행에서 중단된 자신의 tmp 정리
                            shutil.rmtree(tmp_dir)

                        # NAS → pid별 tmp 복사 (NFS blocking 구간)
                        shutil.copytree(nas_scene_dir, tmp_dir)

                        try:
                            # atomic rename: tmp → 최종 경로
                            os.rename(tmp_dir, local_scene_dir)
                        except OSError:
                            # 다른 워커가 먼저 rename 성공 → local_scene_dir 이미 존재
                            # 자신의 tmp만 정리하고 그 결과물을 사용
                            if os.path.exists(tmp_dir):
                                shutil.rmtree(tmp_dir)
                            # local_scene_dir이 이미 있으므로 그대로 진행

                    except Exception:
                        # copytree 실패 (NFS 일시 오류 등) → tmp 정리 후 폴백
                        if os.path.exists(tmp_dir):
                            try:
                                shutil.rmtree(tmp_dir)
                            except Exception:
                                pass
                        if not os.path.isdir(local_scene_dir):
                            # 최종 폴백: local이 없으면 NAS 경로로 직접 읽기
                            # (with 블록 내 return → Lock이 정상 해제됨)
                            return img_path

                # ── LRU 등록 (복사 완료 or 프리페치 데몬이 이미 복사한 경우) ──
                self._scene_cache[scene_id] = local_scene_dir
                self._scene_cache.move_to_end(scene_id)

                # ── LRU 초과 시 가장 오래된 씬 삭제 ─────────────────────
                # max_cache_scenes=9999(carla_e2e_finetune.py)이면 사실상 삭제 없음
                while len(self._scene_cache) > self.max_cache_scenes:
                    oldest_id, oldest_dir = self._scene_cache.popitem(last=False)
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)

        # NAS 경로에서 씬 루트 이후 부분(samples/CAM_FRONT/000036.jpg)을 추출해
        # 로컬 캐시 경로로 재조합한다.
        # 예) img_path = /mnt/nas/.../scene_0065/samples/CAM_FRONT/000036.jpg
        #     → rel = samples/CAM_FRONT/000036.jpg
        #     → local = /tmp/carla_cache/scene_0065/samples/CAM_FRONT/000036.jpg
        idx = img_path.index(scene_id)
        rel_path = img_path[idx + len(scene_id):].lstrip(os.sep)
        return os.path.join(local_scene_dir, rel_path)

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results['img_filename']
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args['backend'] == 'petrel':
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args['backend'] == 'disk':
                if self.local_cache_dir:
                    # ── 로컬 캐시 활성화: NAS → 로컬 복사 후 읽기 ────
                    # 씬 단위로 한 번만 복사되므로 이후 동일 씬의 이미지는
                    # 로컬 SSD에서 읽혀 NFS blocking이 발생하지 않는다.
                    img_path = self._ensure_scene_cached(img_path)
                img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            #[mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f"local_cache_dir='{self.local_cache_dir}', "
        repr_str += f'max_cache_scenes={self.max_cache_scenes})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 with_future_anns=False,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
                 **kwargs):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d

        self.ins_inds_add_1 = ins_inds_add_1
    
    def _load_future_anns(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        # gt_valid_flags = []
        gt_vis_tokens  = []

        for ann_info in results['occ_future_ann_infos']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
                gt_labels_3d.append(ann_info['gt_labels_3d'])
                
                ann_gt_inds = ann_info['gt_inds']
                if self.ins_inds_add_1:
                    ann_gt_inds += 1
                    # NOTE: sdc query is changed from -10 -> -9
                gt_inds_3d.append(ann_gt_inds)

                # gt_valid_flags.append(ann_info['gt_valid_flag'])
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                # invalid frame
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                # gt_valid_flags.append(None)
                gt_vis_tokens.append(None)

        results['future_gt_bboxes_3d'] = gt_bboxes_3d
        # results['future_bbox3d_fields'].append('gt_bboxes_3d')  # Field is used for augmentations, not needed here
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        # results['future_gt_valid_flag'] = gt_valid_flags
        results['future_gt_vis_tokens'] = gt_vis_tokens

        return results 
  
    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results['ann_info']['gt_inds'].copy()

        # NOTE: Avoid gt_inds generated twice
        results['ann_info'].pop('gt_inds')
        
        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results['gt_inds'] = ann_gt_inds
        return results

    def __call__(self, results):
        results = super().__call__(results)
        
        if self.with_future_anns:
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)
        
        # Generate ann for plan
        if 'occ_future_ann_infos_for_plan' in results.keys():
            results = self._load_future_anns_plan(results)
        
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_future_anns={self.with_future_anns}, '
        repr_str += f'{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, '
        
        return repr_str