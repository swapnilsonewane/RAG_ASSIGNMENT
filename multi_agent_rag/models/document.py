from sqlalchemy import (
    Column,
    String,
    DateTime,
    UniqueConstraint,
    Index,
)
from sqlalchemy.sql import func

from multi_agent_rag.core.database import Base


class IngestedDocument(Base):
    __tablename__ = "ingested_documents"

    id = Column(String(36), primary_key=True)

    user_id = Column(String(64), nullable=False, index=True)

    file_hash = Column(String(64), nullable=False)
    filename = Column(String(255), nullable=False)


    status = Column(
        String(32),
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (

        UniqueConstraint(
            "user_id",
            "file_hash",
            name="uq_user_filehash",
        ),

        Index(
            "idx_ingested_documents_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "idx_ingested_documents_user_status",
            "user_id",
            "status",
        ),
    )
