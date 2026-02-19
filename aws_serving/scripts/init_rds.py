#!/usr/bin/env python3
"""
Initialize RDS PostgreSQL database - creates sentiment_db and tables
Run this ONCE before deploying the backend
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Your RDS connection details
RDS_HOST = "sentiment-db.cv2iq0sos7cc.ap-south-2.rds.amazonaws.com"
RDS_PORT = 5432
MASTER_USER = "sentiment_user"
MASTER_PASSWORD = input("Enter your RDS password: ").strip()

print(f"\n🔗 Connecting to RDS at {RDS_HOST}...")

try:
    # Step 1: Connect to default 'postgres' database
    conn = psycopg2.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        database='postgres',
        user=MASTER_USER,
        password=MASTER_PASSWORD,
        sslmode='require',  # AWS RDS requires SSL
        connect_timeout=10
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    print("✅ Connected to RDS")
    
    # Step 2: Check if sentiment_db exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname='sentiment_db'")
    exists = cur.fetchone()
    
    if exists:
        print("ℹ️  Database 'sentiment_db' already exists")
    else:
        print("📦 Creating database 'sentiment_db'...")
        cur.execute("CREATE DATABASE sentiment_db")
        print("✅ Database 'sentiment_db' created")
    
    cur.close()
    conn.close()
    
    # Step 3: Connect to sentiment_db and create tables
    print("\n📊 Creating tables in sentiment_db...")
    conn = psycopg2.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        database='sentiment_db',
        user=MASTER_USER,
        password=MASTER_PASSWORD,
        sslmode='require',
        connect_timeout=10
    )
    cur = conn.cursor()
    
    # Create predictions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_text TEXT NOT NULL,
            prediction VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            model_version VARCHAR(50),
            response_time_ms INTEGER,
            user_ip VARCHAR(50),
            session_id VARCHAR(100)
        );
    """)
    
    # Create indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON predictions(prediction);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON predictions(model_type);")
    
    conn.commit()
    print("✅ Table 'predictions' created with indexes")
    
    # Verify
    cur.execute("SELECT COUNT(*) FROM predictions;")
    count = cur.fetchone()[0]
    print(f"📈 Current prediction count: {count}")
    
    cur.close()
    conn.close()
    
    print("\n🎉 Database initialization complete!")
    print(f"\n📝 Use this DATABASE_URL in your docker-compose.yml:")
    print(f"   postgresql://{MASTER_USER}:{MASTER_PASSWORD}@{RDS_HOST}:5432/sentiment_db")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    raise
