import sqlite3
import uuid

class Database:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        # Optional: Increase performance, but slightly higher risk on crash
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA journal_mode = WAL;") # Write-Ahead Logging often helps concurrency reads/writes

    def create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS pages_raw (
          page_id     TEXT PRIMARY KEY,
          user_id     TEXT NOT NULL,
          doc_id      TEXT NOT NULL,
          page_index  INTEGER NOT NULL,
          hash        TEXT NOT NULL,
          s3_key      TEXT NOT NULL,
          dedup_id    TEXT
        );
        CREATE TABLE IF NOT EXISTS pages_dedup (
          dedup_id  TEXT PRIMARY KEY,
          hash      TEXT UNIQUE NOT NULL
        );
        """)
        self.conn.commit()

    def insert_raw(self, page_id, user_id, doc_id, page_index, h, s3_key):
        self.conn.execute("""
          INSERT OR IGNORE INTO pages_raw
            (page_id,user_id,doc_id,page_index,hash,s3_key)
          VALUES (?,?,?,?,?,?);
        """, (page_id,user_id,doc_id,page_index,h,s3_key))

    def commit(self):
        """Explicitly commit the transaction."""
        self.conn.commit()

    def populate_dedup(self):
        """Build pages_dedup and backfill pages_raw.dedup_id."""
        cur = self.conn.cursor()
        # insert unique hashes
        cur.execute("SELECT DISTINCT hash FROM pages_raw;")
        hashes_to_insert = []
        for (h,) in cur:
            dedup_id = str(uuid.uuid4())
            hashes_to_insert.append((dedup_id, h))

        # Use executemany for potentially faster inserts
        cur.executemany("""
            INSERT OR IGNORE INTO pages_dedup(dedup_id,hash)
            VALUES (?,?);
        """, hashes_to_insert)

        # update raw.dedup_id
        cur.execute("""
          UPDATE pages_raw
             SET dedup_id =
               (SELECT dedup_id FROM pages_dedup WHERE pages_dedup.hash = pages_raw.hash)
           WHERE dedup_id IS NULL;
        """)
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.commit() # Ensure any pending changes are saved
            self.conn.close()