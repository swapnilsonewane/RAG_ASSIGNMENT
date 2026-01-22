from sqlalchemy import Column, String, DateTime
from datetime import datetime

from multi_agent_rag.core.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    role = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
