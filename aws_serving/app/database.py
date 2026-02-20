"""
database.py — PostgreSQL Database Integration
=============================================
SQLAlchemy models and session management for prediction logging.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Get database URL from environment variable
# Format: postgresql://username:password@host:port/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db"
)

# SSL mode for RDS connections
SSL_MODE = os.getenv("DATABASE_SSL_MODE", "prefer")  # prefer, require, or disable

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10,
    connect_args={"sslmode": SSL_MODE} if "rds.amazonaws.com" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Model for storing prediction logs"""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    input_text = Column(Text, nullable=False)
    prediction = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    model_type = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50))
    response_time_ms = Column(Integer)
    user_ip = Column(String(50))
    session_id = Column(String(100))

    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, confidence={self.confidence})>"


def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("[database] ✅ Database tables initialized")
    except Exception as e:
        print(f"[database] ❌ Failed to initialize database: {e}")


def get_db():
    """Get database session (generator for dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get database session (direct)"""
    return SessionLocal()
