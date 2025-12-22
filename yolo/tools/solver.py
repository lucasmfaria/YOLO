from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler
from yolo.utils.model_utils import apply_transfer_freeze
from PIL import Image
import numpy as _np
import torch as _torch


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Apply transfer-learning freezing per config: freeze first N or unfreeze last N.
        transfer_cfg = getattr(cfg.task, "transfer", None) or getattr(cfg, "transfer", None)
        try:
            freeze_n = int(getattr(transfer_cfg, "freeze", 0)) if transfer_cfg is not None else 0
            unfreeze_n = int(getattr(transfer_cfg, "unfreeze_last", 0)) if transfer_cfg is not None else 0
        except Exception:
            freeze_n, unfreeze_n = 0, 0

        result = apply_transfer_freeze(self.model, freeze_first=freeze_n, unfreeze_last=unfreeze_n, layer_attr="model")
        # Ensure at least one parameter is trainable
        if result.get("unfrozen", 0) == 0:
            raise ValueError(f"No trainable parameters after applying transfer config: freeze={freeze_n}, unfreeze_last={unfreeze_n}")
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        if getattr(self.cfg.task, "return_detections", None):  # returns cropped detection images as PIL Images
            img_list = self._crop_detections(origin_frame, predicts)
            return img_list, fps
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")

    def _crop_detections(self, origin_frame, predicts):
        """Crop detected boxes from the original image(s).

        Args:
            origin_frame: PIL.Image or numpy array or list/tuple of images corresponding to the batch.
            predicts: A prediction object or a list of prediction dicts. Each prediction dict is expected
                to contain a box tensor/array under one of the common keys: 'boxes', 'bboxes', or 'bbox',
                and a label under 'labels' or 'classes'. Boxes are expected in xyxy format.

        Returns:
            List[dict]: Each dict has keys 'image' (PIL.Image cropped region) and 'class' (label name).
        """
        results = []

        # Normalize origin_frame to list for easier handling
        if isinstance(origin_frame, (list, tuple)):
            origin_images = list(origin_frame)
        else:
            origin_images = [origin_frame]

        # Normalize predicts to list
        predict_list = predicts if isinstance(predicts, (list, tuple)) else [predicts]

        for img_idx, pred in enumerate(predict_list):
            # Select the matching origin image (if fewer provided, reuse the first)
            orig = origin_images[img_idx] if img_idx < len(origin_images) else origin_images[0]

            # Convert origin to PIL.Image if needed
            if isinstance(orig, _np.ndarray):
                pil_img = Image.fromarray(orig)
            elif hasattr(orig, "cpu") and isinstance(orig, _torch.Tensor):
                arr = orig.detach().cpu().numpy()
                # If tensor shape is (C,H,W) convert to HWC
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = _np.transpose(arr, (1, 2, 0))
                pil_img = Image.fromarray(arr.astype(_np.uint8))
            elif isinstance(orig, Image.Image):
                pil_img = orig
            else:
                # Fallback: try to convert via numpy
                try:
                    arr = _np.asarray(orig)
                    pil_img = Image.fromarray(arr.astype(_np.uint8))
                except Exception:
                    continue


            # Predictions from bbox_nms are typically a Tensor [N x 6]:
            # [class, x1, y1, x2, y2, conf]. The post-process also may return a dict/list.
            boxes_arr = None
            labels_arr = None

            # If prediction is a dict with conventional keys
            if isinstance(pred, dict):
                for key in ("boxes", "bboxes", "bbox"):
                    if key in pred:
                        boxes_arr = pred[key]
                        break
                for key in ("labels", "classes", "class"):
                    if key in pred:
                        labels_arr = pred[key]
                        break

            # If prediction is a tensor/ndarray with rows [cls, x1, y1, x2, y2, conf]
            if boxes_arr is None and isinstance(pred, (_torch.Tensor, _np.ndarray)):
                pred_arr = pred
                if isinstance(pred_arr, _torch.Tensor):
                    pred_arr = pred_arr.detach().cpu().numpy()
                pred_arr = _np.asarray(pred_arr)
                if pred_arr.size == 0:
                    continue
                # Extract boxes and labels
                if pred_arr.shape[1] >= 5:
                    boxes_arr = pred_arr[:, 1:5]
                    labels_arr = pred_arr[:, 0]
                else:
                    # Unexpected shape, skip
                    continue

            # Convert boxes/labels to numpy arrays if they are tensors or lists
            if boxes_arr is None:
                continue
            if hasattr(boxes_arr, "cpu") and isinstance(boxes_arr, _torch.Tensor):
                boxes_arr = boxes_arr.detach().cpu().numpy()
            else:
                boxes_arr = _np.asarray(boxes_arr)

            if labels_arr is None:
                labels_arr = [None] * len(boxes_arr)
            else:
                if hasattr(labels_arr, "cpu") and isinstance(labels_arr, _torch.Tensor):
                    labels_arr = labels_arr.detach().cpu().numpy()
                else:
                    labels_arr = _np.asarray(labels_arr)

            # For each detection, crop and add to results
            for i, box in enumerate(boxes_arr):
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                except Exception:
                    continue

                # Clamp coordinates to image bounds
                w, h = pil_img.size
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = pil_img.crop((x1, y1, x2, y2))

                # Resolve class name
                cls_name = None
                label_val = labels_arr[i] if len(labels_arr) > i else None
                if label_val is None:
                    cls_name = None
                else:
                    # If label is numeric index, map via dataset class list when available
                    try:
                        idx = int(label_val)
                        cls_list = getattr(self.cfg.dataset, "class_list", None)
                        if cls_list and 0 <= idx < len(cls_list):
                            cls_name = cls_list[idx]
                        else:
                            cls_name = str(idx)
                    except Exception:
                        cls_name = str(label_val)

                results.append({"image": cropped, "class": cls_name})

        return results
