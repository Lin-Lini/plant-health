# app/schemas.py
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class SpeciesTopK(BaseModel):
    latin: str
    russian: str
    prob: float

class PlantItem(BaseModel):
    id: int
    cls: str
    conf: float
    bbox: BBox
    area: int
    tilt_deg: float
    dry_ratio: float
    species: Optional[str] = None
    species_latin: Optional[str] = None
    species_russian: Optional[str] = None
    species_conf: float = 0.0
    species_topk: List[SpeciesTopK] = Field(default_factory=list)
    mask_rle: Optional[str] = None
    health_score: float = 100.0
    health_grade: Optional[str] = None

class DefectItem(BaseModel):
    id: int
    cls: str
    conf: float
    bbox: BBox
    area: int
    plant_id: Optional[int] = None
    mask_rle: Optional[str] = None
    area_ratio: float = 0.0
    severity: Optional[str] = None

class InferResponse(BaseModel):
    request_id: str
    status: Literal["OK", "NO_PLANTS", "ERROR"] = Field(..., description="OK | NO_PLANTS | ERROR")
    overlay_path: Optional[str] = None
    report_json_path: Optional[str] = None
    plants: List[PlantItem] = Field(default_factory=list)
    defects: List[DefectItem] = Field(default_factory=list)
    # extras с произвольными типами: int/bool/dict — как в твоём примере
    extras: Dict[str, Any] = Field(default_factory=dict)