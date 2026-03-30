from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParsedElement(BaseModel):
    element_id: str
    type: str
    text: str
    page_number: Optional[int] = None
    category_depth: Optional[int] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    manual_id: str
    filename: str
    content_type: str
    element_count: int
    elements: List[ParsedElement]
