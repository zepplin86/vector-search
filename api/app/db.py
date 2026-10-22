from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from . import config

pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=8, configure=register_vector, open=False)


def open_pool() -> None:
    pool.open()


def close_pool() -> None:
    pool.close()


def set_status(video_id: str, kind: str, status: str, count: int | None = None, error: str | None = None) -> None:
    col_status = f"{kind}_status"
    col_count = f"{kind}_count"
    with pool.connection() as conn:
        conn.execute(
            f"""
            INSERT INTO videos (id, {col_status}, {col_count}, error, updated_at)
            VALUES (%s, %s, COALESCE(%s, 0), %s, now())
            ON CONFLICT (id) DO UPDATE SET
              {col_status} = EXCLUDED.{col_status},
              {col_count}  = COALESCE(%s, videos.{col_count}),
              error        = %s,
              updated_at   = now()
            """,
            (video_id, status, count, error, count, error),
        )


def set_duration(video_id: str, duration: float) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO videos (id, duration) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET duration = EXCLUDED.duration",
            (video_id, duration),
        )
