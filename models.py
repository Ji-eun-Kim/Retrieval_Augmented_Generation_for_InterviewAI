from sqlalchemy import Column, Integer, String, Text

from database import Base


class RAG(Base):
    __tablename__ = 'rag'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    grade = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    feedback = Column(Text, nullable=True)


class LLM(Base):
    __tablename__ = 'llm'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    grade = Column(String, nullable=False)
    reason = Column(String, nullable=False)
