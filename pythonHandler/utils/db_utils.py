import os
import logging
import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)

# Global connection pool
connection_pool = None


def init_connection_pool():
    """Initialize the connection pool"""
    global connection_pool

    try:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not found")

        # Parse connection parameters from URL
        conn_params = {}

        # Simple URL parser for postgres://user:pass@host:port/dbname
        if database_url.startswith("postgres://"):
            # Remove postgres://
            url = database_url[11:]

            # Split user:pass@host:port/dbname
            userpass_host_port_dbname = url.split("@")

            if len(userpass_host_port_dbname) > 1:
                userpass = userpass_host_port_dbname[0].split(":")
                hostport_dbname = userpass_host_port_dbname[1].split("/")

                # Set connection parameters
                conn_params["user"] = userpass[0]
                if len(userpass) > 1:
                    conn_params["password"] = userpass[1]

                hostport = hostport_dbname[0].split(":")
                conn_params["host"] = hostport[0]
                if len(hostport) > 1:
                    conn_params["port"] = hostport[1]

                if len(hostport_dbname) > 1:
                    conn_params["dbname"] = hostport_dbname[1]
        else:
            # Just use the URL directly
            conn_params["dsn"] = database_url

        # Create connection pool with 5 connections
        connection_pool = psycopg2.pool.ThreadedConnectionPool(1, 5, **conn_params)
        logger.info("Database connection pool initialized")

        return connection_pool
    except Exception as e:
        logger.error(f"Error initializing connection pool: {str(e)}")
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
