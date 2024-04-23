import logging
import os
from dotenv import load_dotenv
import psycopg2

from psycopg2.extensions import (
    ISOLATION_LEVEL_AUTOCOMMIT,
)  # <-- Needed for creating a database


logger = logging.getLogger(__name__)

# Assuming addaterm is used elsewhere in your code not shown here


load_dotenv()

PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")


def pg_connect(
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    host=PG_HOST,
):
    """
    Connect to the domains  database and return the connection object.

    :param database: Name of the database to connect to.
    :param user: Username for authentication.
    :param password: Password for authentication.
    :param host: Database host address.
    :return: Connection object.
    """
    try:
        conn = psycopg2.connect(
            dbname=database, user=user, password=password, host=host
        )
        return conn
    except psycopg2.Error as e:
        logger.info(f"Error connecting to PostgreSQL database: {e}")
        raise e


def sum_query_info():
    conn = pg_connect()
    if not conn:
        logger.info("Could not connect to database")
        return
    cursor = conn.cursor()

    query = """
    SELECT
    COUNT(*) FILTER (WHERE status IS NULL OR status = 'ready') AS count_ready_or_null,
    COUNT(*) FILTER (WHERE status = 'finished') AS count_finished,
    COUNT(*) FILTER (WHERE status = 'in_process') AS count_in_process
    FROM pdf_links;
    """

    cursor.execute(query)
    results = cursor.fetchall()
    print(f"There are {results[0][0]} pdf_links ready to be processed")
    print(f"There are {results[0][1]} pdf_links are already finished")
    print(f"There are {results[0][2]} pdf_links are in process right now")
    cursor.close()
    conn.close()
    return


def fetchPdfLinks(batch_size):
    """
    Fetches a batch of PDF links that are ready for processing.
    Marks fetched links as 'in_process'.

    :param connection: A psycopg2 connection object to the database.
    :param batch_size: The number of rows to fetch.
    :return: A list of dictionaries representing the fetched rows.
    """
    connection = pg_connect()
    with connection.cursor() as cursor:
        # Atomically select and update rows
        try:
            cursor.execute(
                """
                WITH selected AS (
                    SELECT id
                    FROM pdf_links
                    WHERE status = 'ready' AND classify IS NULL
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE pdf_links
                SET status = 'in_process'
                FROM selected
                WHERE pdf_links.id = selected.id
                RETURNING pdf_links.*
            """,
                (batch_size,),
            )

            rows = cursor.fetchall()
            connection.commit()
            # Convert the rows to a list of dictionaries
            colnames = [desc[0] for desc in cursor.description]
            return [dict(zip(colnames, row)) for row in rows]
        except psycopg2.Error as e:
            logger.info(f"Database error: {e}")
            if connection:
                connection.rollback()
            return []


def reset_to_ready(ids, table="domains"):
    # Ensure ids are integers to prevent SQL injection
    ids = [int(x) for x in ids]  # Assuming IDs are always integers

    # Create a string with placeholders for each ID
    placeholders = ", ".join(["%s"] * len(ids))

    conn = pg_connect()
    if not conn:
        logger.warning("Failed to connect to the database")
        return

    try:
        cursor = conn.cursor()
        query = f"""UPDATE {table} SET status = 'ready' WHERE id IN ({placeholders}) AND status = 'in_process';"""
        cursor.execute(query, ids)
        count = cursor.rowcount
        if count > 0:
            logger.info(f"{count} {table}s are being restored to ready status.")
        conn.commit()
    except Exception as e:
        logger.info(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def return_to_ready():
    # Return all links marked in_process back to ready. A maintenance method
    conn = pg_connect()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE pdf_links SET status = 'ready', last_updated = CURRENT_TIMESTAMP WHERE status = 'in_process';"
        )
        conn.commit()
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        logger.info(f"Database error: {e}")
        if conn:
            conn.rollback()


def update_link(id, classify, failure=False):
    conn = pg_connect()
    if not conn:
        return
    try:
        id = int(id)
        classify = int(classify)
        cursor = conn.cursor()
        if failure:
            cursor.execute(
                "UPDATE pdf_links SET status = 'ready', failure='Failed to download', last_updated = CURRENT_TIMESTAMP WHERE id = %s;",
                (id,),
            )
        else:
            cursor.execute(
                "UPDATE pdf_links SET classify = %s, status = 'finished', last_updated = CURRENT_TIMESTAMP WHERE id = %s;",
                (classify, id),
            )
        conn.commit()
        cursor.close()
        conn.close()
        return id
    except psycopg2.Error as e:
        logger.info(f"Database error: {e}")
        if conn:
            conn.rollback()
        return -1
        


if __name__ == "__main__":
    # links = fetchPdfLinks(10)
    # lll = [x['id'] for x in links] 
    # print(lll)
    return_to_ready()
    # ids = [97030, 97246, 99986, 100116, 100701, 144373]
    # reset_to_ready(ids, table="pdf_links")

    sum_query_info()
    # return_to_ready()  # [row[0] for row in process])
    # sum_query_info()
