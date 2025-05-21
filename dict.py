"""
@author: Mbaka JohnPaul
"""

from pydantic import BaseModel

class Recommendation1(BaseModel):
    mealTime: str
    
class Recommendation(BaseModel):
    mealTime: str
    item: str
