# app/models/loaders.py
from typing import Any, Dict, Optional, Tuple, List
import os
from glob import glob
import cv2
import torch
from ultralytics import YOLO

from app import config

def _pick_weight(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

def _first_in_dirs(dirs: List[str]) -> Optional[str]:
    exts = ("*.pt", "*.pth", "*.onnx")
    for d in dirs:
        for e in exts:
            found = sorted(glob(os.path.join(d, e)))
            if found:
                return found[0]
    return None

class LazyModels:
    """
    Ленивые загрузчики моделей:
      - YOLO сегментация растений/дефектов
      - Глубина: сначала Depth Anything (если доступен), затем фоллбек на MiDaS
      - (Опционально) species_cls через ONNX, но в текущем пайплайне используется TorchScript (см. species_infer)
    """
    def __init__(self) -> None:
        self._plant = None
        self._defect = None
        self.models = {}

    # ----------------------- YOLO SEG -----------------------

    def _resolve_yolo_weights(self, path: Optional[str], default_name: str = "yolov8n-seg.pt") -> str:
        p = (path or "").strip()
        return p if (p and os.path.exists(p)) else default_name

    def _load_yolo_seg(self, weights: str):
        model = YOLO(weights)
        # fuse уменьшает накладные расходы на инференс
        model.fuse()
        return model

    def plant_seg(self):
        if self._plant is None:
            # 1) явный путь из переменной окружения/конфига
            explicit = _pick_weight([getattr(config, "PLANT_SEG_WEIGHTS", None)])
            # 2) поиск в стандартных папках внутри образа
            discovered = _first_in_dirs(["/srv/app/weights/plant", "/weights/plant"])
            path = explicit or discovered
            if path:
                print(f"[loaders] Using PLANT weights: {path}")
                self._plant = YOLO(path)
            else:
                print("[loaders] PLANT weights not found, using fallback yolov8n-seg.pt")
                self._plant = YOLO("yolov8n-seg.pt")
        return self._plant

    def defect_seg(self):
        if self._defect is None:
            explicit = _pick_weight([getattr(config, "DEFECT_SEG_WEIGHTS", None)])
            discovered = _first_in_dirs(["/srv/app/weights/defect", "/weights/defect"])
            path = explicit or discovered
            if path:
                print(f"[loaders] Using DEFECT weights: {path}")
                self._defect = YOLO(path)
            else:
                print("[loaders] DEFECT weights not found, using fallback yolov8n-seg.pt")
                self._defect = YOLO("yolov8n-seg.pt")
        return self._defect

    # ----------------------- DEPTH -----------------------

    def depth_anything(self):
        """
        Пытаемся инициализировать Depth Anything.
        Кэшируем результат: ("depth_anything", model) или None.
        """
        if "depth" in self.models:
            return self.models["depth"]

        if os.getenv("DEPTH_DISABLE", "0") == "1":
            self.models["depth"] = None
            return None

        w = os.getenv("DEPTH_ANYTHING_WEIGHTS", "").strip()
        try:
            from depth_anything.dpt import DepthAnything
            # Если указан путь к локальным весам — используем его, иначе hub-имя
            src = w if (w and os.path.isdir(w)) else "LiheYoung/depth_anything_vitl14"
            m = DepthAnything.from_pretrained(src).eval().to(config.DEVICE)
            self.models["depth"] = ("depth_anything", m)
        except Exception:
            # Если не удалось (нет пакета/весов/сети), пробуем MiDaS уже в depth_map_for()
            self.models["depth"] = None
        return self.models["depth"]

    def _get_midas(self) -> Optional[Tuple[Any, Any]]:
        """
        Лениво поднимаем MiDaS + transforms. Кэшируем ("midas", (midas, transforms)).
        Возвращаем кортеж (midas, transforms) или None.
        """
        if "midas" in self.models:
            return self.models["midas"]

        try:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
            midas.to(config.DEVICE).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.models["midas"] = (midas, transforms)
            return self.models["midas"]
        except Exception:
            self.models["midas"] = None
            return None

    def depth_map_for(self, image_bgr):
        """
        Возвращает карту глубины float32 в диапазоне 0..1 с размером (H, W)
        или None, если глубина отключена/недоступна.
        Приоритет: Depth Anything -> MiDaS (фоллбек).
        """
        if os.getenv("DEPTH_DISABLE", "0") == "1":
            return None

        # 1) Depth Anything
        item = self.depth_anything()
        if isinstance(item, tuple) and item and item[0] == "depth_anything":
            _, m = item
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(config.DEVICE) / 255.0
            with torch.inference_mode():
                d = m(t)[0, 0]  # (h', w')
            d = d.detach().float().cpu().numpy()
            H, W = image_bgr.shape[:2]
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            return d.astype("float32")

        # 2) MiDaS фоллбек
        mid = self._get_midas()
        if mid is None:
            return None

        midas, transforms = mid
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transform = transforms.dpt_transform  # соответствует DPT_Hybrid
        input_batch = transform(img_rgb).to(config.DEVICE)

        with torch.inference_mode():
            pred = midas(input_batch)  # (1, H', W') тензор
            # масштабируем до исходного разрешения
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze(0).squeeze(0)

        d = pred.float().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        return d.astype("float32")

    # ----------------------- SPECIES (не используется в текущем пайплайне) -----------------------

    def _load_species(self, weights: str):
        """
        Опциональный ONNX-вариант для species-классификации.
        В текущем пайплайне используется TorchScript (см. app/models/species_infer.py).
        """
        if weights and os.path.exists(weights):
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
                return ("onnx", sess)
            except Exception:
                return ("stub", None)
        return ("stub", None)

    def species_cls(self):
        if "species_cls" not in self.models:
            self.models["species_cls"] = self._load_species(config.SPECIES_CLS_WEIGHTS)
        return self.models["species_cls"]

    # ----------------------- Сохранённый бэккомпат -----------------------

    def depth_model(self):
        """
        Бэккомпат-метод, если кто-то извне ожидает .depth_model().
        Возвращает то же, что и depth_anything/_get_midas в совокупности.
        """
        if "depth" not in self.models:
            # сначала Depth Anything
            self.depth_anything()
        if self.models.get("depth") is None:
            # затем MiDaS (для совместимости вернём "midas", если он есть)
            self._get_midas()
        return self.models.get("depth", None)