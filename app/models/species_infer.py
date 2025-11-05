# app/models/species_infer.py
import os, json, logging
from glob import glob
from typing import List, Dict, Any, Optional

import torch
import torchvision.transforms as T
from PIL import Image

from app import config

log = logging.getLogger("species")

# ленивые синглтоны
_TS = None
_CLASSES: List[str] | None = None
_RU: Dict[str, str] = {}

# препроцесс
_IMG = int(os.getenv("SPECIES_IMG_SIZE", "384"))
_TF = T.Compose([T.Resize((_IMG, _IMG)), T.ToTensor()])

# где искать файлы по умолчанию (как у plant/defect)
_SPECIES_DIRS = [
    "/srv/app/models/species",
    "/models/species",
    "models/species",
    "/srv/app/weights/species",
    "/weights/species",
]

def _resolve_file(env_path: str | None,
                  basenames: List[str],
                  search_dirs: List[str]) -> Optional[str]:
    """Выбираем первый существующий файл среди:
    1) явного пути из env (и его вариант, склеенный с CWD для относительных),
    2) комбинаций search_dir + base.
    """
    candidates: List[str] = []
    p = (env_path or "").strip()
    if p:
        candidates.append(p)
        if not os.path.isabs(p):
            candidates.append(os.path.join(os.getcwd(), p))

    for d in search_dirs:
        for b in basenames:
            candidates.append(os.path.join(d, b))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None

def _lazy_load() -> None:
    global _TS, _CLASSES, _RU
    if _TS is not None:
        return

    # 1) TorchScript
    ts_path = _resolve_file(
        os.getenv("SPECIES_TS", config.SPECIES_TS),
        basenames=["model_ts.pt", "species_ts.pt", "species_model.pt"],
        search_dirs=_SPECIES_DIRS,
    )
    if not ts_path:
        log.warning(f"[species] TorchScript not found. Tried env/default and {_SPECIES_DIRS}")
        _TS = False
    else:
        try:
            _TS = torch.jit.load(ts_path, map_location=config.DEVICE).eval()
            log.info(f"[species] TorchScript loaded: {ts_path}")
        except Exception as e:
            log.exception(f"[species] failed to load TorchScript {ts_path}: {e}")
            _TS = False

    # 2) classes.json
    _CLASSES = None
    classes_path = _resolve_file(
        os.getenv("SPECIES_CLASSES", config.SPECIES_CLASSES),
        basenames=["species_classes.json", "classes.json"],
        search_dirs=_SPECIES_DIRS,
    )
    if classes_path:
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                _CLASSES = data
                log.info(f"[species] classes loaded: {classes_path} (n={len(_CLASSES)})")
            else:
                log.error(f"[species] {classes_path} is not list[str]")
        except Exception as e:
            log.exception(f"[species] failed to read {classes_path}: {e}")
    else:
        log.warning("[species] classes file not found (optional).")

    # 3) ru_map.json
    _RU = {}
    ru_path = _resolve_file(
        os.getenv("SPECIES_RU_MAP", config.SPECIES_RU_MAP),
        basenames=["species_ru_map.json", "ru_map.json"],
        search_dirs=_SPECIES_DIRS,
    )
    if ru_path:
        try:
            with open(ru_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict):
                _RU = {str(k): str(v) for k, v in m.items()}
                log.info(f"[species] ru_map loaded: {ru_path} (n={len(_RU)})")
            else:
                log.error(f"[species] {ru_path} is not dict")
        except Exception as e:
            log.exception(f"[species] failed to read {ru_path}: {e}")
    else:
        log.warning("[species] ru_map file not found (optional).")

@torch.inference_mode()
def predict_species(pil_image: Image.Image, topk: int = 3) -> List[Dict[str, Any]]:
    _lazy_load()
    if _TS is False:
        return []

    x = _TF(pil_image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    logits = _TS(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()
    idx = prob.argsort()[::-1][:max(1, int(topk))]

    out: List[Dict[str, Any]] = []
    for i in idx:
        latin = None
        if _CLASSES and 0 <= int(i) < len(_CLASSES):
            latin = _CLASSES[int(i)]
        name_latin = latin if latin else f"class_{int(i)}"
        name_ru = _RU.get(name_latin, name_latin)
        out.append({"latin": name_latin, "russian": name_ru, "prob": float(prob[int(i)])})
    return out