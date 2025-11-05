import os

DEVICE = os.getenv("DEVICE", "cuda:0")
THRESH_PLANT = float(os.getenv("THRESH_PLANT", "0.25"))
THRESH_DEFECT = float(os.getenv("THRESH_DEFECT", "0.01"))

PLANT_CLASSES = [x.strip() for x in os.getenv("PLANT_CLASSES", "tree,shrub").split(",") if x.strip()]
PLANT_COLORS  = [x.strip() for x in os.getenv("PLANT_COLORS", "").split(",") if x.strip()]
DEFECT_COLORS = [x.strip() for x in os.getenv("DEFECT_COLORS", "").split(",") if x.strip()]

PLANT_SEG_WEIGHTS = os.getenv("PLANT_SEG_WEIGHTS", "/srv/app/weights/plant/plant_seg.pt").strip()
DEFECT_SEG_WEIGHTS = os.getenv("DEFECT_SEG_WEIGHTS", "/srv/app/weights/defect/defect_seg.pt").strip()

# Species TS + словари. По умолчанию — внутри образа.
SPECIES_TS       = os.getenv("SPECIES_TS", "/srv/app/models/species/model_ts.pt").strip()
SPECIES_CLASSES  = os.getenv("SPECIES_CLASSES", "/srv/app/models/species/species_classes").strip()
SPECIES_RU_MAP   = os.getenv("SPECIES_RU_MAP", "/srv/app/models/species/species_ru_map").strip()
SPECIES_STUB_LABEL = os.getenv("SPECIES_STUB_LABEL", "unknown")
SPECIES_IMG_SIZE = int(os.getenv("SPECIES_IMG_SIZE", "384"))

PLANT_MERGE_BBOX_IOU  = 0.3
PLANT_MERGE_CONTAIN   = 0.5
PLANT_MERGE_IOU       = 0.5

SEVERITY_RULES_CSV = os.getenv("SEVERITY_RULES_CSV", "/srv/app/rules/severity_rules.csv").strip()
OUT_DIR = os.getenv("OUT_DIR", "/data/out")

PLANT_FALLBACK  = os.getenv("PLANT_FALLBACK", "0") == "1"
DEFECT_FALLBACK = os.getenv("DEFECT_FALLBACK", "0") == "1"

VEG_H_MIN = int(os.getenv("VEG_H_MIN", "30"))
VEG_H_MAX = int(os.getenv("VEG_H_MAX", "90"))
VEG_S_MIN = int(os.getenv("VEG_S_MIN", "40"))
VEG_V_MIN = int(os.getenv("VEG_V_MIN", "40"))

BRN_H_MIN = int(os.getenv("BRN_H_MIN", "10"))
BRN_H_MAX = int(os.getenv("BRN_H_MAX", "25"))
BRN_S_MIN = int(os.getenv("BRN_S_MIN", "20"))
BRN_V_MAX = int(os.getenv("BRN_V_MAX", "180"))