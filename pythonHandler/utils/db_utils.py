"""Utility functions for database interactions."""

import os
import logging
import psycopg2
from psycopg2 import pool
from urllib.parse import urlparse
from utils.prisma_client import prisma_client

logger = logging.getLogger(__name__)

# Global connection pool
connection_pool = None


def init_connection_pool():
    """Initialize the connection pool"""
    global connection_pool
    database_url = None  # Ensure database_url is defined for the except block

    try:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            logger.critical(
                "DATABASE_URL environment variable not found. Cannot initialize connection pool."
            )
            raise ValueError("DATABASE_URL environment variable not found")

        conn_params = {"dsn": database_url}

        log_dsn = "DSN (details in env)"  # Default safe log
        if database_url:
            try:
                parsed = urlparse(database_url)
                if parsed.password:
                    # Construct the safe DSN carefully, avoiding complex f-string internals
                    scheme = parsed.scheme if parsed.scheme else "postgres"
                    user = parsed.username if parsed.username else "user"
                    host = parsed.hostname if parsed.hostname else "host"
                    port_str = f":{parsed.port}" if parsed.port else ""
                    path = parsed.path if parsed.path else "/dbname"
                    query = f"?{parsed.query}" if parsed.query else ""
                    log_dsn = (
                        f"{scheme}://{user}:********@{host}{port_str}{path}{query}"
                    )
                else:
                    log_dsn = database_url  # No password, log as is
            except Exception:
                log_dsn = "DSN (error redacting, original in env)"  # Fallback on parsing error

        logger.info(f"Initializing connection pool with DSN: {log_dsn}...")

        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=5, **conn_params
        )
        logger.info("Database connection pool initialized successfully using DSN.")

        return connection_pool
    except ValueError as ve:
        # This will be caught if DATABASE_URL is None
        logger.critical(str(ve))  # Log the original ValueError
        raise ve
    except Exception as e:
        log_dsn_err = "DSN not available or error before DSN logging (DATABASE_URL might be missing)"
        if database_url:  # Check if database_url was fetched before the error
            log_dsn_err = "DSN (details in env, error during init)"  # Generic message if URL was present
            # Avoid complex parsing here to prevent further syntax errors in error path
            # The original error 'e' is more important.

        logger.error(
            f"Error initializing connection pool. DSN used (potentially unsafe if error was in redaction): '{database_url if database_url else 'NOT_SET'}'. Error: {str(e)}"
        )
        raise


def get_connection_pool():
    """Get the connection pool, initializing it if needed"""
    global connection_pool

    if connection_pool is None:
        connection_pool = init_connection_pool()

    return connection_pool


def get_connection():
    """Get a connection from the pool"""
    pool = get_connection_pool()
    return pool.getconn()


def release_connection(conn):
    """Release a connection back to the pool"""
    pool = get_connection_pool()
    pool.putconn(conn)


def execute_query(query, params=None, fetch=True):
    """Execute a query using a connection from the pool"""
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(query, params)

        if fetch:
            results = cursor.fetchall()
            return results
        else:
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_connection(conn)


def validate_api_key(api_key: str, track_usage: bool = True) -> str:
    """Validate API key and return user ID if valid.

    Args:
        api_key: API key to validate
        track_usage: Whether to track API usage for this validation

    Returns:
        str: User ID if valid, empty string if invalid
    """
    if not api_key:
        logger.error("Empty API key provided")
        return ""

    try:
        # Clean the API key
        api_key = api_key.strip()

        # Use Prisma client to find user by API key
        user = prisma_client.get_user_by_api_key(api_key)

        if not user:
            logger.error("Invalid API key provided")
            return ""

        # Log the found user data (excluding sensitive info)
        logger.info(f"API key validated for userId: {str(user['id'])}")

        if not user["id"]:
            logger.error("User data found but missing id")
            return ""

        # Track API usage on successful validation if requested
        if track_usage:
            try:
                prisma_client.track_api_usage(api_key)
            except Exception as tracking_error:
                # Don't fail the validation if tracking fails
                logger.warning(f"Error tracking API usage: {tracking_error}")

        return user["id"]

    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return ""


def update_process_status(status_text: str, mode: str, user_id: str) -> None:
    """Log process status without database updates."""
    # NOTE: This function currently only logs. If DB updates are needed,
    # they should be added here using prisma_client.
    try:
        logger.info(
            f"Process status update - mode: {mode}, user: {user_id}, status: {status_text}"
        )
    except Exception as e:
        logger.error(f"Error logging process status: {e}")
        logger.error(
            f"Status update attempted - mode: {mode}, user: {user_id}, status: {status_text}"
        )
