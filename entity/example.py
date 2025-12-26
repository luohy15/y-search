from sqlalchemy import Column, Integer, String, Text
from entity.base import BaseEntity

class Example(BaseEntity):
    """Example table model"""
    __tablename__ = 'example'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=True)