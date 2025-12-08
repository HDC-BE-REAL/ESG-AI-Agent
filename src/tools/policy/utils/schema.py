from pydantic import BaseModel
from typing import List, Optional

class PolicySection(BaseModel):
    title: str
    content: str

class PolicyDocument(BaseModel):
    sections: List[PolicySection]
    requirements: List[str] = []
