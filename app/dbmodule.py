import psycopg2
import os

def connect():
    """Connect to the Supabase/Postgres DB."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )
    return conn

def get_all_attendance():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, timestamp, confidence, detection_method, verified FROM attendance_records;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id": r[0],
            "student_id": r[1],
            "timestamp": r[2],
            "confidence": r[3],
            "detection_method": r[4],
            "verified": r[5]
        } for r in rows
    ]


def get_attendance_by_student(student_id):
    """Fetch attendance records for a specific student."""
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM attendance_records WHERE student_id = %s;",
        (str(student_id).strip(),)  # convert to string and strip spaces
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id": r[0],
            "student_id": str(r[1]),
            "status": str(r[2]).strip() if r[2] is not None else "",
            "created_at": str(r[3]),
            "verified": r[4]
        }
        for r in rows
    ]
