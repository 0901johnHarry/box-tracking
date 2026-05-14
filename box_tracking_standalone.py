import argparse
import csv
import math
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


DEFAULT_MODEL = "best_26m.pt"
DEFAULT_SOURCE = "record_20260211_104117.mkv"
DEFAULT_ROI = (1436.0, 450.0, 3833.0, 1133.0)
DEFAULT_CLASS_ID = 1
DEFAULT_CROP_TO_ROI = True
DEFAULT_CROP_PAD = 80


class BoxKalmanFilter:
    def __init__(self, bbox_corners, max_area_change_rate=0.02, display_smooth_factor=0.4):
        cx, cy, w, h = self._corners_to_params(bbox_corners)

        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:, 4:] *= 0.02
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(-1, 1)

        self.id = None
        self.age = 0
        self.hits = 1
        self.consecutive_hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.confidence = 1.0
        self.history = [self._params_to_corners(cx, cy, w, h)]
        self.partial_offset = None
        self.is_partial = False
        self.matched_det_corners = None
        self.update_state = "TENTATIVE"
        self.size_lock_state = "UNLOCKED"
        self.last_detection_area = w * h
        self.last_visible_ratio = 1.0
        self.last_center_offset_ratio = 0.0

        self.max_area_change_rate = float(max_area_change_rate)
        self.reference_area = w * h
        self.last_frame_area = w * h
        self.area_locked = False
        self.stable_w = max(float(w), 10.0)
        self.stable_h = max(float(h), 10.0)
        self.stable_area = self.stable_w * self.stable_h
        self.stable_aspect = self.stable_w / max(self.stable_h, 1.0)
        self.stable_update_alpha = 0.03
        self.min_stable_update_visible_ratio = 0.90
        self.max_stable_aspect_diff = 0.25
        self.display_smooth_factor = float(display_smooth_factor)
        self.smooth_corners = self._params_to_corners(cx, cy, w, h)

    @staticmethod
    def _corners_to_params(corners):
        corners = np.asarray(corners, dtype=np.float32)
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        return (x_min + x_max) * 0.5, (y_min + y_max) * 0.5, x_max - x_min, y_max - y_min

    @staticmethod
    def _params_to_corners(cx, cy, w, h):
        half_w = w * 0.5
        half_h = h * 0.5
        return np.array(
            [
                [cx - half_w, cy - half_h],
                [cx + half_w, cy - half_h],
                [cx + half_w, cy + half_h],
                [cx - half_w, cy + half_h],
            ],
            dtype=np.float32,
        )

    def lock_constraints(self):
        state = self.kf.statePost.ravel()
        self.stable_w = max(float(state[2]), 10.0)
        self.stable_h = max(float(state[3]), 10.0)
        self.stable_area = self.stable_w * self.stable_h
        self.stable_aspect = self.stable_w / max(self.stable_h, 1.0)
        self.reference_area = self.stable_area
        self.last_frame_area = self.reference_area
        self.area_locked = True
        self.size_lock_state = "LOCKED"
        self._force_state_stable_size()
        state = self.kf.statePost.ravel()
        self.smooth_corners = self._params_to_corners(state[0], state[1], self.stable_w, self.stable_h)

    def ready_to_lock_stable_size(
        self,
        min_hits=8,
        window=5,
        max_total_growth_rate=0.08,
        max_step_growth_rate=0.05,
        max_aspect_jitter=0.12,
        force_after_hits=24,
    ):
        if self.area_locked:
            return True
        if self.consecutive_hits < min_hits or len(self.history) < window:
            self.size_lock_state = "ENTERING"
            return False

        recent = self.history[-window:]
        params = [self._corners_to_params(corners) for corners in recent]
        widths = np.array([max(p[2], 10.0) for p in params], dtype=np.float32)
        heights = np.array([max(p[3], 10.0) for p in params], dtype=np.float32)
        areas = widths * heights
        aspects = widths / np.maximum(heights, 1.0)

        total_growth = (areas[-1] - areas[0]) / (areas[0] + 1e-6)
        step_growth = np.diff(areas) / (areas[:-1] + 1e-6)
        max_step_growth = float(np.max(step_growth)) if len(step_growth) else 0.0
        aspect_jitter = float((aspects.max() - aspects.min()) / (aspects.mean() + 1e-6))

        still_growing = total_growth > max_total_growth_rate or max_step_growth > max_step_growth_rate
        aspect_unstable = aspect_jitter > max_aspect_jitter
        if still_growing or aspect_unstable:
            self.size_lock_state = "ENTERING"
            return self.consecutive_hits >= force_after_hits

        self.size_lock_state = "READY"
        return True

    def _force_state_stable_size(self):
        if not self.area_locked:
            return
        state = self.kf.statePost.ravel()
        state[2] = self.stable_w
        state[3] = self.stable_h
        state[6] = 0.0
        state[7] = 0.0
        self.kf.statePost = state.reshape(-1, 1)

    def _maybe_update_stable_size(self, measured_w, measured_h, visible_ratio):
        if not self.area_locked:
            return
        measured_w = max(float(measured_w), 10.0)
        measured_h = max(float(measured_h), 10.0)
        measured_aspect = measured_w / max(measured_h, 1.0)
        aspect_diff = abs(measured_aspect - self.stable_aspect) / max(self.stable_aspect, 1e-6)
        if visible_ratio < self.min_stable_update_visible_ratio:
            return
        if aspect_diff > self.max_stable_aspect_diff:
            return
        alpha = self.stable_update_alpha
        self.stable_w = self.stable_w * (1.0 - alpha) + measured_w * alpha
        self.stable_h = self.stable_h * (1.0 - alpha) + measured_h * alpha
        self.stable_area = self.stable_w * self.stable_h
        self.stable_aspect = self.stable_w / max(self.stable_h, 1.0)
        self.reference_area = self.stable_area
        self.last_frame_area = self.stable_area

    def _clamp_state(self):
        if not self.area_locked:
            return

        state = self.kf.statePost.ravel()
        current_w = max(state[2], 10)
        current_h = max(state[3], 10)
        current_area = current_w * current_h
        change_rate = (current_area - self.last_frame_area) / (self.last_frame_area + 1e-6)

        if abs(change_rate) > self.max_area_change_rate:
            target_area = self.last_frame_area * (
                1.0 + self.max_area_change_rate if change_rate > 0 else 1.0 - self.max_area_change_rate
            )
            scale = math.sqrt(target_area / (current_area + 1e-6))
            state[2] = current_w * scale
            state[3] = current_h * scale
            state[6] = 0.0
            state[7] = 0.0
            current_area = target_area
            self.kf.statePost = state.reshape(-1, 1)

        self.last_frame_area = current_area

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self._clamp_state()
        self._force_state_stable_size()
        self.matched_det_corners = None
        state = self.kf.statePost.ravel()
        return self._params_to_corners(state[0], state[1], max(state[2], 10), max(state[3], 10))

    @staticmethod
    def corners_area(corners):
        corners = np.asarray(corners, dtype=np.float32)
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)

    def _record_debug_metrics(self, corners, update_state, detection_area=None, visible_ratio=None, center_offset_ratio=None):
        self.update_state = update_state
        if detection_area is not None:
            self.last_detection_area = float(detection_area)
        else:
            self.last_detection_area = self.corners_area(corners)
        if visible_ratio is not None:
            self.last_visible_ratio = float(visible_ratio)
        else:
            self.last_visible_ratio = self.last_detection_area / (self.reference_area + 1e-6)
        if center_offset_ratio is not None:
            self.last_center_offset_ratio = float(center_offset_ratio)

    def soft_update(self, corners, detection_area=None, visible_ratio=None, center_offset_ratio=None, update_stable_size=False):
        self._record_debug_metrics(
            corners,
            "FULL",
            detection_area=detection_area,
            visible_ratio=visible_ratio,
            center_offset_ratio=center_offset_ratio,
        )
        cx, cy, w, h = self._corners_to_params(corners)
        if self.area_locked and not update_stable_size:
            measurement = np.array([cx, cy, self.stable_w, self.stable_h], dtype=np.float32).reshape(-1, 1)
        else:
            measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement)
        self._clamp_state()
        if update_stable_size:
            self._maybe_update_stable_size(w, h, self.last_visible_ratio)
        self._force_state_stable_size()
        self.time_since_update = 0
        self.hits += 1
        self.consecutive_hits += 1
        self.confidence = min(1.0, self.hits / (self.age + 1e-6))
        self.partial_offset = None
        self.is_partial = False
        self._save_history()
        return self.get_state()

    def partial_position_update(self, corners, detection_area=None, visible_ratio=None, center_offset_ratio=None):
        self._record_debug_metrics(
            corners,
            "PARTIAL",
            detection_area=detection_area,
            visible_ratio=visible_ratio,
            center_offset_ratio=center_offset_ratio,
        )
        det_cx, det_cy, _, _ = self._corners_to_params(corners)
        state = self.kf.statePost.ravel()
        pred_cx, pred_cy = state[0], state[1]

        if not self.is_partial:
            self.partial_offset = (det_cx - pred_cx, det_cy - pred_cy)
            self.is_partial = True
            self.time_since_update = 0
            self.hits += 1
            self.consecutive_hits += 1
            return self.get_state()

        est_cx = det_cx - self.partial_offset[0]
        est_cy = det_cy - self.partial_offset[1]
        measurement = np.array([est_cx, est_cy, state[2], state[3]], dtype=np.float32).reshape(-1, 1)
        original_r = self.kf.measurementNoiseCov.copy()
        self.kf.measurementNoiseCov[0, 0] = 3.0
        self.kf.measurementNoiseCov[1, 1] = 3.0
        self.kf.measurementNoiseCov[2, 2] = 1000.0
        self.kf.measurementNoiseCov[3, 3] = 1000.0
        self.kf.correct(measurement)
        self.kf.measurementNoiseCov = original_r
        self._clamp_state()
        self._force_state_stable_size()
        self.time_since_update = 0
        self.hits += 1
        self.consecutive_hits += 1
        self.confidence = min(1.0, self.hits / (self.age + 1e-6))
        self._save_history()
        return self.get_state()

    def mark_not_validated(self):
        if self.confirmed:
            self.update_state = "PREDICT"
        else:
            self.update_state = "TENTATIVE"
        self.consecutive_hits = 0
        state = self.kf.statePost.ravel()
        state[4] *= 0.95
        state[5] *= 0.95
        state[6] *= 0.95
        state[7] *= 0.95
        self.kf.statePost = state.reshape(-1, 1)

    def get_raw_display_corners(self):
        kalman_corners = self.get_state()
        if self.area_locked:
            state = self.kf.statePost.ravel()
            return self._params_to_corners(state[0], state[1], self.stable_w, self.stable_h)
        if self.matched_det_corners is None:
            return kalman_corners
        all_det_pts = np.vstack(self.matched_det_corners)
        det_x_min = np.min(all_det_pts[:, 0])
        det_y_min = np.min(all_det_pts[:, 1])
        det_x_max = np.max(all_det_pts[:, 0])
        det_y_max = np.max(all_det_pts[:, 1])
        return np.array(
            [
                [min(kalman_corners[0][0], det_x_min), min(kalman_corners[0][1], det_y_min)],
                [max(kalman_corners[1][0], det_x_max), min(kalman_corners[1][1], det_y_min)],
                [max(kalman_corners[2][0], det_x_max), max(kalman_corners[2][1], det_y_max)],
                [min(kalman_corners[3][0], det_x_min), max(kalman_corners[3][1], det_y_max)],
            ],
            dtype=np.float32,
        )

    def get_display_corners(self):
        raw = self.get_raw_display_corners()
        alpha = self.display_smooth_factor
        self.smooth_corners = self.smooth_corners * (1.0 - alpha) + raw * alpha
        return self.smooth_corners.copy()

    def _save_history(self):
        self.history.append(self.get_state().copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def get_state(self):
        state = self.kf.statePost.ravel()
        return self._params_to_corners(state[0], state[1], max(state[2], 10), max(state[3], 10))

    def get_params(self):
        state = self.kf.statePost.ravel()
        if self.area_locked:
            return {"cx": state[0], "cy": state[1], "w": self.stable_w, "h": self.stable_h, "vx": state[4], "vy": state[5]}
        return {"cx": state[0], "cy": state[1], "w": state[2], "h": state[3], "vx": state[4], "vy": state[5]}

    def get_area(self):
        if self.area_locked:
            return self.stable_area
        state = self.kf.statePost.ravel()
        return max(state[2], 10) * max(state[3], 10)

    def get_constraint_info(self):
        if not self.area_locked:
            return {
                "area_vs_ref": 1.0,
                "ref_area": self.get_area(),
                "detection_area": self.last_detection_area,
                "visible_ratio": self.last_visible_ratio,
                "center_offset_ratio": self.last_center_offset_ratio,
            }
        return {
            "area_vs_ref": self.get_area() / (self.reference_area + 1e-6),
            "ref_area": self.reference_area,
            "stable_w": self.stable_w,
            "stable_h": self.stable_h,
            "stable_aspect": self.stable_aspect,
            "detection_area": self.last_detection_area,
            "visible_ratio": self.last_visible_ratio,
            "center_offset_ratio": self.last_center_offset_ratio,
        }


class MaskCornerExtractor:
    def __init__(self, min_area=500):
        self.min_area = float(min_area)

    def extract_corners_from_mask(self, mask):
        if mask is None or mask.sum() == 0:
            return None
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)
        sum_xy = pts[:, 0] + pts[:, 1]
        diff_xy = pts[:, 0] - pts[:, 1]
        diff_yx = pts[:, 1] - pts[:, 0]
        corners = np.array(
            [
                pts[np.argmin(sum_xy)],
                pts[np.argmax(diff_xy)],
                pts[np.argmax(sum_xy)],
                pts[np.argmax(diff_yx)],
            ],
            dtype=np.float32,
        )
        if self._quad_area(corners) < self.min_area:
            return None
        return corners

    @staticmethod
    def _quad_area(corners):
        x, y = corners[:, 0], corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def extract_corners_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


class PredictionDrivenTracker:
    def __init__(
        self,
        max_age=20,
        confirm_frames=3,
        area_ratio_full=0.75,
        area_ratio_partial=0.30,
        false_det_distance_ratio=1.5,
        max_area_change_rate=0.02,
        display_smooth_factor=0.4,
        stable_lock_min_hits=8,
        stable_lock_window=5,
        stable_lock_max_total_growth_rate=0.08,
        stable_lock_max_step_growth_rate=0.05,
        stable_lock_max_aspect_jitter=0.12,
        stable_lock_force_after_hits=24,
    ):
        self.max_age = int(max_age)
        self.confirm_frames = int(confirm_frames)
        self.area_ratio_full = float(area_ratio_full)
        self.area_ratio_partial = float(area_ratio_partial)
        self.false_det_distance_ratio = float(false_det_distance_ratio)
        self.max_area_change_rate = float(max_area_change_rate)
        self.display_smooth_factor = float(display_smooth_factor)
        self.stable_lock_min_hits = int(stable_lock_min_hits)
        self.stable_lock_window = int(stable_lock_window)
        self.stable_lock_max_total_growth_rate = float(stable_lock_max_total_growth_rate)
        self.stable_lock_max_step_growth_rate = float(stable_lock_max_step_growth_rate)
        self.stable_lock_max_aspect_jitter = float(stable_lock_max_aspect_jitter)
        self.stable_lock_force_after_hits = int(stable_lock_force_after_hits)
        self.trackers = []
        self.next_id = 1
        self.frame_count = 0

    def _try_lock_stable_size(self, tracker):
        if tracker.area_locked:
            return
        if tracker.ready_to_lock_stable_size(
            min_hits=self.stable_lock_min_hits,
            window=self.stable_lock_window,
            max_total_growth_rate=self.stable_lock_max_total_growth_rate,
            max_step_growth_rate=self.stable_lock_max_step_growth_rate,
            max_aspect_jitter=self.stable_lock_max_aspect_jitter,
            force_after_hits=self.stable_lock_force_after_hits,
        ):
            tracker.lock_constraints()

    @staticmethod
    def _corners_to_bbox_batch(corners_list):
        if not corners_list:
            return np.empty((0, 4), dtype=np.float32)
        arr = np.array(corners_list)
        return np.column_stack(
            [arr[:, :, 0].min(axis=1), arr[:, :, 1].min(axis=1), arr[:, :, 0].max(axis=1), arr[:, :, 1].max(axis=1)]
        )

    @staticmethod
    def _overlap_matrix(bboxes1, bboxes2):
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.empty((len(bboxes1), len(bboxes2)), dtype=np.float32)
        x1 = np.maximum(bboxes1[:, 0:1], bboxes2[:, 0:1].T)
        y1 = np.maximum(bboxes1[:, 1:2], bboxes2[:, 1:2].T)
        x2 = np.minimum(bboxes1[:, 2:3], bboxes2[:, 2:3].T)
        y2 = np.minimum(bboxes1[:, 3:4], bboxes2[:, 3:4].T)
        return np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    @staticmethod
    def _iou_matrix(bboxes1, bboxes2):
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.empty((len(bboxes1), len(bboxes2)), dtype=np.float32)
        inter = PredictionDrivenTracker._overlap_matrix(bboxes1, bboxes2)
        a1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        a2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)

    @staticmethod
    def _center_offset_ratio(candidate_corners, tracker):
        candidate_center = np.asarray(candidate_corners, dtype=np.float32).mean(axis=0)
        tracker_center = tracker.get_state().mean(axis=0)
        state = tracker.kf.statePost.ravel()
        scale = max(float(state[2]), float(state[3]), 1.0)
        return float(np.linalg.norm(candidate_center - tracker_center) / scale)

    def update(self, detection_corners_list):
        self.frame_count += 1
        for tracker in self.trackers:
            tracker.predict()

        used_dets = set()
        det_bboxes = self._corners_to_bbox_batch(detection_corners_list)
        det_areas = np.zeros(len(detection_corners_list), dtype=np.float32)
        if len(det_bboxes) > 0:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (det_bboxes[:, 3] - det_bboxes[:, 1])

        confirmed = [t for t in self.trackers if t.confirmed]
        if confirmed and detection_corners_list:
            pred_bboxes = self._corners_to_bbox_batch([t.get_state() for t in confirmed])
            pred_areas = np.array([t.get_area() for t in confirmed])
            overlap_ratio = self._overlap_matrix(pred_bboxes, det_bboxes) / (det_areas[None, :] + 1e-6)
            for ti, tracker in enumerate(confirmed):
                overlapping_indices = [i for i in np.where(overlap_ratio[ti] > 0.3)[0] if i not in used_dets]
                if not overlapping_indices:
                    tracker.mark_not_validated()
                    continue
                for idx in overlapping_indices:
                    used_dets.add(idx)
                tracker.matched_det_corners = [detection_corners_list[i] for i in overlapping_indices]
                all_pts = np.vstack(tracker.matched_det_corners)
                x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
                x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()
                combined = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                combined_area = (x_max - x_min) * (y_max - y_min)
                area_ratio = combined_area / (pred_areas[ti] + 1e-6)
                center_offset_ratio = self._center_offset_ratio(combined, tracker)
                if area_ratio > self.area_ratio_full:
                    tracker.soft_update(
                        combined,
                        detection_area=combined_area,
                        visible_ratio=area_ratio,
                        center_offset_ratio=center_offset_ratio,
                        update_stable_size=area_ratio >= tracker.min_stable_update_visible_ratio,
                    )
                    self._try_lock_stable_size(tracker)
                elif area_ratio > self.area_ratio_partial:
                    tracker.partial_position_update(
                        combined,
                        detection_area=combined_area,
                        visible_ratio=area_ratio,
                        center_offset_ratio=center_offset_ratio,
                    )
        else:
            for tracker in confirmed:
                tracker.mark_not_validated()

        unconfirmed = [t for t in self.trackers if not t.confirmed]
        remaining = [i for i in range(len(detection_corners_list)) if i not in used_dets]
        if unconfirmed and remaining:
            rem_bboxes = self._corners_to_bbox_batch([detection_corners_list[i] for i in remaining])
            unc_bboxes = self._corners_to_bbox_batch([t.get_state() for t in unconfirmed])
            cost = 1.0 - self._iou_matrix(rem_bboxes, unc_bboxes)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_ti = set()
            for di, ti in zip(row_ind, col_ind):
                if cost[di, ti] < 0.7:
                    det_idx = remaining[di]
                    tracker = unconfirmed[ti]
                    tracker.matched_det_corners = [detection_corners_list[det_idx]]
                    det_area = BoxKalmanFilter.corners_area(detection_corners_list[det_idx])
                    tracker.soft_update(
                        detection_corners_list[det_idx],
                        detection_area=det_area,
                        visible_ratio=1.0,
                        center_offset_ratio=self._center_offset_ratio(detection_corners_list[det_idx], tracker),
                    )
                    used_dets.add(det_idx)
                    matched_ti.add(ti)
                    if tracker.consecutive_hits >= self.confirm_frames:
                        tracker.confirmed = True
                        self._try_lock_stable_size(tracker)
            for ti, tracker in enumerate(unconfirmed):
                if ti not in matched_ti:
                    tracker.mark_not_validated()
        else:
            for tracker in unconfirmed:
                tracker.mark_not_validated()

        self._create_new_trackers([i for i in range(len(detection_corners_list)) if i not in used_dets], detection_corners_list)
        self._prune_dead_trackers()
        return self._build_results()

    def _create_new_trackers(self, new_indices, detection_corners_list):
        if not new_indices:
            return
        active = [t for t in self.trackers if t.time_since_update <= 5]
        for i in new_indices:
            if active:
                det_center = detection_corners_list[i].mean(axis=0)
                active_centers = np.array([t.get_state().mean(axis=0) for t in active])
                active_max_dims = np.array([max(t.kf.statePost.ravel()[2], t.kf.statePost.ravel()[3]) for t in active])
                if np.any(np.linalg.norm(active_centers - det_center[None, :], axis=1) < active_max_dims * self.false_det_distance_ratio):
                    continue
            tracker = BoxKalmanFilter(
                detection_corners_list[i],
                max_area_change_rate=self.max_area_change_rate,
                display_smooth_factor=self.display_smooth_factor,
            )
            tracker.id = self.next_id
            self.next_id += 1
            tracker.matched_det_corners = [detection_corners_list[i]]
            tracker.update_state = "TENTATIVE"
            self.trackers.append(tracker)

    def _prune_dead_trackers(self):
        alive = []
        for tracker in self.trackers:
            if tracker.confirmed and tracker.time_since_update <= self.max_age:
                alive.append(tracker)
            elif not tracker.confirmed and tracker.time_since_update <= 5:
                alive.append(tracker)
        self.trackers = alive

    def _build_results(self):
        results = []
        for tracker in self.trackers:
            if not tracker.confirmed:
                continue
            results.append(
                {
                    "id": tracker.id,
                    "corners": tracker.get_display_corners(),
                    "kalman_corners": tracker.get_state(),
                    "params": tracker.get_params(),
                    "confidence": tracker.confidence,
                    "age": tracker.age,
                    "time_since_update": tracker.time_since_update,
                    "is_partial": tracker.is_partial,
                    "constraints": tracker.get_constraint_info(),
                    "has_detection": tracker.matched_det_corners is not None,
                    "id_source": "PredictionTracker",
                    "state": tracker.update_state,
                    "debug": {
                        "detection_area": tracker.last_detection_area,
                        "stable_area": tracker.reference_area if tracker.area_locked else tracker.get_area(),
                        "stable_w": tracker.stable_w if tracker.area_locked else tracker.get_params()["w"],
                        "stable_h": tracker.stable_h if tracker.area_locked else tracker.get_params()["h"],
                        "stable_aspect": (
                            tracker.stable_aspect
                            if tracker.area_locked
                            else tracker.get_params()["w"] / max(tracker.get_params()["h"], 1.0)
                        ),
                        "size_lock_state": tracker.size_lock_state,
                        "visible_ratio": tracker.last_visible_ratio,
                        "center_offset_ratio": tracker.last_center_offset_ratio,
                    },
                }
            )
        return results


class BoxTrackingSystem:
    def __init__(
        self,
        model_path,
        conf_threshold=0.5,
        roi=None,
        target_class_id=None,
        max_area_change_rate=0.02,
        display_smooth_factor=0.4,
        crop_pad=50,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = float(conf_threshold)
        self.roi = roi
        self.target_class_id = target_class_id
        self.crop_pad = int(crop_pad)
        self.extractor = MaskCornerExtractor()
        self.tracker = PredictionDrivenTracker(
            max_area_change_rate=max_area_change_rate,
            display_smooth_factor=display_smooth_factor,
        )
        self.roi_pixel = None
        self.frame_initialized = False
        self.crop_x1 = 0
        self.crop_y1 = 0
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

    def _init_roi(self, frame_h, frame_w):
        if self.roi is None:
            return
        x1, y1, x2, y2 = self.roi
        if all(0 <= v <= 1.0 for v in self.roi):
            x1, y1, x2, y2 = int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)
        self.roi_pixel = (max(0, int(x1)), max(0, int(y1)), min(frame_w, int(x2)), min(frame_h, int(y2)))

    def _is_in_roi(self, corners):
        if self.roi_pixel is None:
            return True
        center = corners.mean(axis=0)
        rx1, ry1, rx2, ry2 = self.roi_pixel
        return rx1 <= center[0] <= rx2 and ry1 <= center[1] <= ry2

    def process_frame(self, frame):
        if not self.frame_initialized:
            self._init_roi(frame.shape[0], frame.shape[1])
            self.frame_initialized = True

        roi_frame = frame
        self.crop_x1, self.crop_y1 = 0, 0
        if self.roi_pixel is not None:
            rx1, ry1, rx2, ry2 = self.roi_pixel
            self.crop_x1 = max(0, rx1 - self.crop_pad)
            self.crop_y1 = max(0, ry1 - self.crop_pad)
            crop_x2 = min(frame.shape[1], rx2 + self.crop_pad)
            crop_y2 = min(frame.shape[0], ry2 + self.crop_pad)
            roi_frame = frame[self.crop_y1:crop_y2, self.crop_x1:crop_x2]

        results = self.model(roi_frame, conf=self.conf_threshold, verbose=False)
        detections = self._extract_detections(results, roi_frame)
        tracks = self.tracker.update(detections)
        tracks = self._filter_roi_results(tracks)
        annotated = self._draw(frame.copy(), detections, tracks)
        return annotated, tracks

    def crop_view(self, frame, pad=80):
        if self.roi_pixel is None:
            return frame
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.roi_pixel
        x1 = max(0, int(rx1) - int(pad))
        y1 = max(0, int(ry1) - int(pad))
        x2 = min(w, int(rx2) + int(pad))
        y2 = min(h, int(ry2) + int(pad))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2].copy()

    def _extract_detections(self, results, roi_frame):
        detections = []
        if len(results) <= 0:
            return detections
        result = results[0]
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for i, mask in enumerate(masks):
                if self.target_class_id is not None and int(classes[i]) != self.target_class_id:
                    continue
                rh, rw = roi_frame.shape[:2]
                if mask.shape[0] != rh or mask.shape[1] != rw:
                    mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
                corners = self.extractor.extract_corners_from_mask((mask * 255).astype(np.uint8))
                if corners is None:
                    continue
                corners[:, 0] += self.crop_x1
                corners[:, 1] += self.crop_y1
                if self._is_in_roi(corners):
                    detections.append(corners)
            return detections

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for i, bbox in enumerate(boxes):
                if self.target_class_id is not None and int(classes[i]) != self.target_class_id:
                    continue
                adjusted = bbox.copy()
                adjusted[0] += self.crop_x1
                adjusted[1] += self.crop_y1
                adjusted[2] += self.crop_x1
                adjusted[3] += self.crop_y1
                corners = self.extractor.extract_corners_from_bbox(adjusted)
                if self._is_in_roi(corners):
                    detections.append(corners)
        return detections

    def _filter_roi_results(self, tracks):
        if self.roi_pixel is None:
            return tracks
        rx1, ry1, rx2, ry2 = self.roi_pixel
        margin = 50
        return [
            t for t in tracks
            if rx1 - margin <= t["corners"].mean(axis=0)[0] <= rx2 + margin
            and ry1 - margin <= t["corners"].mean(axis=0)[1] <= ry2 + margin
        ]

    def _draw(self, frame, detections, tracks):
        h, w = frame.shape[:2]
        if self.roi_pixel is not None:
            rx1, ry1, rx2, ry2 = self.roi_pixel
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2, cv2.LINE_AA)

        for corners in detections:
            cv2.polylines(frame, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 180, 0), 1, cv2.LINE_AA)

        for track in tracks:
            track_id = track["id"]
            corners = track["corners"]
            params = track["params"]
            debug = track.get("debug", {})
            color = self.colors[track_id % len(self.colors)]
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            center = corners.mean(axis=0).astype(np.int32)
            state = track.get("state") or ("DETECT" if track["has_detection"] else "PREDICT")
            lines = [
                f"ID:{track_id} {state} src:{track.get('id_source', 'PredictionTracker')}",
                f"det:{int(bool(track['has_detection']))} miss:{track['time_since_update']}",
                f"W:{params['w']:.0f} H:{params['h']:.0f} vis:{debug.get('visible_ratio', 0.0):.2f} off:{debug.get('center_offset_ratio', 0.0):.2f}",
                f"detA:{debug.get('detection_area', 0.0):.0f} refA:{debug.get('stable_area', 0.0):.0f}",
                f"stable:{debug.get('stable_w', params['w']):.0f}x{debug.get('stable_h', params['h']):.0f} lock:{debug.get('size_lock_state', 'NA')}",
            ]
            tx = max(0, min(center[0] - 120, w - 260))
            ty = max(58, center[1] - 34)
            text_width = 250
            line_h = 17
            cv2.rectangle(
                frame,
                (tx - 4, ty - 16),
                (tx + text_width, ty + line_h * (len(lines) - 1) + 6),
                (0, 0, 0),
                -1,
            )
            for idx, text in enumerate(lines):
                cv2.putText(
                    frame,
                    text,
                    (tx, ty + idx * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return frame


def parse_roi(values):
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI needs 4 numbers: x1 y1 x2 y2")
    return tuple(float(v) for v in values)


def main():
    parser = argparse.ArgumentParser(description="Standalone box tracking test runner.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model path for box detection/segmentation.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video path, camera index, or RTSP URL.")
    parser.add_argument("--roi", nargs=4, type=float, default=DEFAULT_ROI, help="ROI: x1 y1 x2 y2. Supports pixels or 0-1 ratios.")
    parser.add_argument("--class-id", type=int, default=DEFAULT_CLASS_ID, help="Target class id. Omit to use all classes.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--output", default=None, help="Optional output video path.")
    parser.add_argument("--debug-log", default=None, help="Optional CSV path for per-frame tracking debug metrics.")
    parser.add_argument("--no-show", action="store_true", help="Disable preview window.")
    parser.add_argument("--full-view", action="store_true", help="Show full frame instead of cropping to ROI.")
    parser.add_argument("--view-pad", type=int, default=DEFAULT_CROP_PAD, help="Extra pixels around ROI when crop view is enabled.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means no limit.")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    tracker = BoxTrackingSystem(
        model_path=args.model,
        conf_threshold=args.conf,
        roi=parse_roi(args.roi),
        target_class_id=args.class_id,
    )

    writer = None
    debug_file = None
    debug_writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = 0
    last_ts = time.perf_counter()
    if args.debug_log:
        debug_file = open(args.debug_log, "w", newline="", encoding="utf-8")
        debug_writer = csv.DictWriter(
            debug_file,
            fieldnames=[
                "frame",
                "track_id",
                "id_source",
                "state",
                "has_detection",
                "time_since_update",
                "x_center",
                "y_center",
                "width",
                "height",
                "detection_area",
                "stable_area",
                "stable_width",
                "stable_height",
                "stable_aspect",
                "size_lock_state",
                "visible_ratio",
                "center_offset_ratio",
            ],
        )
        debug_writer.writeheader()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            annotated, tracks = tracker.process_frame(frame)
            display_frame = annotated
            if DEFAULT_CROP_TO_ROI and not args.full_view:
                display_frame = tracker.crop_view(annotated, pad=args.view_pad)

            if debug_writer is not None:
                for track in tracks:
                    params = track["params"]
                    debug = track.get("debug", {})
                    debug_writer.writerow(
                        {
                            "frame": frame_count,
                            "track_id": track["id"],
                            "id_source": track.get("id_source", "PredictionTracker"),
                            "state": track.get("state"),
                            "has_detection": int(bool(track["has_detection"])),
                            "time_since_update": track["time_since_update"],
                            "x_center": float(params["cx"]),
                            "y_center": float(params["cy"]),
                            "width": float(params["w"]),
                            "height": float(params["h"]),
                            "detection_area": float(debug.get("detection_area", 0.0)),
                            "stable_area": float(debug.get("stable_area", 0.0)),
                            "stable_width": float(debug.get("stable_w", params["w"])),
                            "stable_height": float(debug.get("stable_h", params["h"])),
                            "stable_aspect": float(debug.get("stable_aspect", 0.0)),
                            "size_lock_state": debug.get("size_lock_state", "NA"),
                            "visible_ratio": float(debug.get("visible_ratio", 0.0)),
                            "center_offset_ratio": float(debug.get("center_offset_ratio", 0.0)),
                        }
                    )

            now = time.perf_counter()
            fps_text = 1.0 / max(now - last_ts, 1e-6)
            last_ts = now
            cv2.putText(
                display_frame,
                f"FPS:{fps_text:.1f} Tracks:{len(tracks)} ID:PredictionTracker",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if args.output and writer is None:
                h, w = display_frame.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if writer is not None:
                writer.write(display_frame)

            if not args.no_show:
                cv2.imshow("box_tracking_standalone", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if debug_file is not None:
            debug_file.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
