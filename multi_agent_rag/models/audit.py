from sqlalchemy import Column, String, DateTime, JSON, func, Index

from multi_agent_rag.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)

    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)

    details = Column(
        "metadata",
        JSON,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_created_at", "created_at"),
    )
