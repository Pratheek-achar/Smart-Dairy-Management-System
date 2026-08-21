"""
database.py – SQLite Animal Profile Database
Smart Dairy Livestock Monitoring System

Manages persistent storage for animal profiles, health records, and behaviour logs.
Uses Python's built-in sqlite3 – no extra dependencies required.
"""

import sqlite3
import os
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'smartdairy.db')


# ── Connect helper ─────────────────────────────────────────────────────────────
def get_db():
    """Return a sqlite3 connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads
    return conn


# ── Initialise schema ──────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    conn = get_db()
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS animals (
        uid         TEXT PRIMARY KEY,          -- SDMS-2024-0001
        name        TEXT NOT NULL,
        breed       TEXT DEFAULT 'Unknown',
        age_years   REAL DEFAULT 0,
        gender      TEXT DEFAULT 'Female',
        weight_kg   REAL DEFAULT 0,
        color       TEXT DEFAULT '',
        tag_number  TEXT DEFAULT '',
        notes       TEXT DEFAULT '',
        status      TEXT DEFAULT 'Active',     -- Active | Quarantine | Sold | Deceased
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS health_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        animal_uid      TEXT NOT NULL,
        recorded_at     TEXT NOT NULL,
        temperature     REAL,
        humidity        REAL,
        milk_yield      REAL,
        weight_kg       REAL,
        heart_rate      REAL,
        activity_level  REAL,
        prediction      TEXT,                  -- 'Healthy' | 'Sick'
        confidence      REAL,
        risk_level      TEXT,
        recommendations TEXT,                  -- JSON string
        FOREIGN KEY (animal_uid) REFERENCES animals(uid) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS behaviour_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        animal_uid  TEXT NOT NULL,
        logged_at   TEXT NOT NULL,
        behaviour   TEXT NOT NULL,
        duration_sec REAL DEFAULT 0,
        velocity    REAL DEFAULT 0,
        alert_msg   TEXT DEFAULT '',
        FOREIGN KEY (animal_uid) REFERENCES animals(uid) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS vaccination_schedules (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        animal_uid      TEXT NOT NULL,
        vaccine_name    TEXT NOT NULL,
        scheduled_date  TEXT NOT NULL,          -- YYYY-MM-DD
        administered_date TEXT DEFAULT NULL,
        status          TEXT DEFAULT 'Pending', -- Pending | Completed | Overdue | Skipped
        dose_number     INTEGER DEFAULT 1,
        next_due_date   TEXT DEFAULT NULL,
        administered_by TEXT DEFAULT '',
        notes           TEXT DEFAULT '',
        created_at      TEXT NOT NULL,
        FOREIGN KEY (animal_uid) REFERENCES animals(uid) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS treatments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        animal_uid      TEXT NOT NULL,
        treatment_type  TEXT NOT NULL,          -- Medicine | Vet Visit | Surgery | Deworming | Other
        medicine_name   TEXT DEFAULT '',
        dosage          TEXT DEFAULT '',
        start_date      TEXT NOT NULL,
        end_date        TEXT DEFAULT NULL,
        status          TEXT DEFAULT 'Active',  -- Active | Completed | Cancelled
        vet_name        TEXT DEFAULT '',
        cost            REAL DEFAULT 0,
        notes           TEXT DEFAULT '',
        created_at      TEXT NOT NULL,
        FOREIGN KEY (animal_uid) REFERENCES animals(uid) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    print("[DB] smartdairy.db initialised.")


# ── UID generator ──────────────────────────────────────────────────────────────
def _generate_uid() -> str:
    """Generate next sequential UID like SDMS-2024-0001."""
    year = datetime.datetime.now().year
    conn = get_db()
    row  = conn.execute(
        "SELECT uid FROM animals WHERE uid LIKE ? ORDER BY uid DESC LIMIT 1",
        (f"SDMS-{year}-%",)
    ).fetchone()
    conn.close()

    if row:
        last_num = int(row['uid'].split('-')[-1])
        return f"SDMS-{year}-{last_num+1:04d}"
    return f"SDMS-{year}-0001"


def _now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ══════════════════════════════════════════════════════════════════════════════
#  ANIMAL CRUD
# ══════════════════════════════════════════════════════════════════════════════

def create_animal(data: dict) -> str:
    """Insert a new animal. Returns the generated UID."""
    uid = _generate_uid()
    now = _now()
    conn = get_db()
    conn.execute("""
        INSERT INTO animals
          (uid, name, breed, age_years, gender, weight_kg, color, tag_number, notes, status, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        uid,
        data.get('name', 'Unknown'),
        data.get('breed', 'Unknown'),
        float(data.get('age_years', 0)),
        data.get('gender', 'Female'),
        float(data.get('weight_kg', 0)),
        data.get('color', ''),
        data.get('tag_number', ''),
        data.get('notes', ''),
        data.get('status', 'Active'),
        now, now
    ))
    conn.commit()
    conn.close()
    return uid


def get_all_animals(status_filter: str = None) -> list[dict]:
    """Return all animals, optionally filtered by status."""
    conn = get_db()
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM animals WHERE status=? ORDER BY created_at DESC",
            (status_filter,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM animals ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_animal(uid: str) -> dict | None:
    """Return a single animal by UID."""
    conn = get_db()
    row  = conn.execute("SELECT * FROM animals WHERE uid=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_animal(uid: str, data: dict) -> bool:
    """Update animal fields. Returns True on success."""
    conn = get_db()
    conn.execute("""
        UPDATE animals SET
            name=?, breed=?, age_years=?, gender=?, weight_kg=?,
            color=?, tag_number=?, notes=?, status=?, updated_at=?
        WHERE uid=?
    """, (
        data.get('name'), data.get('breed'),
        float(data.get('age_years', 0)),
        data.get('gender'), float(data.get('weight_kg', 0)),
        data.get('color'), data.get('tag_number'),
        data.get('notes'), data.get('status'),
        _now(), uid
    ))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


def delete_animal(uid: str) -> bool:
    """Delete animal and all related records."""
    conn = get_db()
    conn.execute("DELETE FROM animals WHERE uid=?", (uid,))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH RECORDS
# ══════════════════════════════════════════════════════════════════════════════

def add_health_record(animal_uid: str, data: dict) -> int:
    """Insert a health prediction record. Returns new row ID."""
    import json as _json
    conn = get_db()
    cur  = conn.execute("""
        INSERT INTO health_records
          (animal_uid, recorded_at, temperature, humidity, milk_yield,
           weight_kg, heart_rate, activity_level, prediction, confidence,
           risk_level, recommendations)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        animal_uid, _now(),
        data.get('temperature'), data.get('humidity'),
        data.get('milk_yield'),  data.get('weight_kg'),
        data.get('heart_rate'),  data.get('activity_level'),
        data.get('prediction'),  data.get('confidence'),
        data.get('risk_level'),
        _json.dumps(data.get('recommendations', []))
    ))
    # also update animal weight if provided
    if data.get('weight_kg'):
        conn.execute(
            "UPDATE animals SET weight_kg=?, updated_at=? WHERE uid=?",
            (data['weight_kg'], _now(), animal_uid)
        )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_health_records(animal_uid: str, limit: int = 50) -> list[dict]:
    """Return health records for an animal, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM health_records WHERE animal_uid=? ORDER BY recorded_at DESC LIMIT ?",
        (animal_uid, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  BEHAVIOUR LOGS
# ══════════════════════════════════════════════════════════════════════════════

def add_behaviour_log(animal_uid: str, data: dict) -> int:
    """Log a behaviour event for an animal."""
    conn = get_db()
    cur  = conn.execute("""
        INSERT INTO behaviour_logs
          (animal_uid, logged_at, behaviour, duration_sec, velocity, alert_msg)
        VALUES (?,?,?,?,?,?)
    """, (
        animal_uid, _now(),
        data.get('behaviour', 'Unknown'),
        float(data.get('duration_sec', 0)),
        float(data.get('velocity', 0)),
        data.get('alert_msg', '')
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_behaviour_logs(animal_uid: str, limit: int = 50) -> list[dict]:
    """Return behaviour logs for an animal, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM behaviour_logs WHERE animal_uid=? ORDER BY logged_at DESC LIMIT ?",
        (animal_uid, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  STATS / AGGREGATES
# ══════════════════════════════════════════════════════════════════════════════

def get_animal_stats(uid: str) -> dict:
    """Return aggregate stats for an animal's profile page."""
    conn = get_db()

    health_count = conn.execute(
        "SELECT COUNT(*) as c FROM health_records WHERE animal_uid=?", (uid,)
    ).fetchone()['c']

    sick_count = conn.execute(
        "SELECT COUNT(*) as c FROM health_records WHERE animal_uid=? AND prediction='Sick'",
        (uid,)
    ).fetchone()['c']

    behaviour_count = conn.execute(
        "SELECT COUNT(*) as c FROM behaviour_logs WHERE animal_uid=?", (uid,)
    ).fetchone()['c']

    # Weight trend (last 10 records)
    weight_trend = conn.execute("""
        SELECT recorded_at, weight_kg FROM health_records
        WHERE animal_uid=? AND weight_kg IS NOT NULL
        ORDER BY recorded_at DESC LIMIT 10
    """, (uid,)).fetchall()

    # Behaviour distribution
    beh_dist = conn.execute("""
        SELECT behaviour, COUNT(*) as cnt FROM behaviour_logs
        WHERE animal_uid=? GROUP BY behaviour ORDER BY cnt DESC
    """, (uid,)).fetchall()

    conn.close()

    return {
        'health_count':   health_count,
        'sick_count':     sick_count,
        'healthy_count':  health_count - sick_count,
        'behaviour_count': behaviour_count,
        'weight_trend':   [dict(r) for r in weight_trend][::-1],  # oldest first
        'behaviour_dist': [dict(r) for r in beh_dist],
    }


def get_dashboard_counts() -> dict:
    """Quick counts for the home dashboard."""
    conn = get_db()
    today  = datetime.date.today().isoformat()
    total  = conn.execute("SELECT COUNT(*) FROM animals").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM animals WHERE status='Active'").fetchone()[0]
    # Mark overdue vaccinations automatically
    conn.execute(
        "UPDATE vaccination_schedules SET status='Overdue' WHERE status='Pending' AND scheduled_date < ?",
        (today,)
    )
    conn.commit()
    pending_vax = conn.execute(
        "SELECT COUNT(*) FROM vaccination_schedules WHERE status='Pending'"
    ).fetchone()[0]
    overdue_vax = conn.execute(
        "SELECT COUNT(*) FROM vaccination_schedules WHERE status='Overdue'"
    ).fetchone()[0]
    conn.close()
    return {'total': total, 'active': active,
            'pending_vax': pending_vax, 'overdue_vax': overdue_vax}


# ══════════════════════════════════════════════════════════════════════════════
#  VACCINATION SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════

def add_vaccination(animal_uid: str, data: dict) -> int:
    conn = get_db()
    cur  = conn.execute("""
        INSERT INTO vaccination_schedules
          (animal_uid, vaccine_name, scheduled_date, dose_number,
           next_due_date, administered_by, notes, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        animal_uid,
        data.get('vaccine_name'),
        data.get('scheduled_date'),
        int(data.get('dose_number', 1)),
        data.get('next_due_date') or None,
        data.get('administered_by', ''),
        data.get('notes', ''),
        _now()
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_vaccinations(animal_uid: str = None, status: str = None) -> list[dict]:
    conn = get_db()
    # Auto-mark overdue
    today = datetime.date.today().isoformat()
    conn.execute(
        "UPDATE vaccination_schedules SET status='Overdue' WHERE status='Pending' AND scheduled_date < ?",
        (today,)
    )
    conn.commit()
    q, params = "SELECT v.*, a.name as animal_name FROM vaccination_schedules v JOIN animals a ON v.animal_uid=a.uid", []
    conditions = []
    if animal_uid:
        conditions.append("v.animal_uid=?"); params.append(animal_uid)
    if status:
        conditions.append("v.status=?"); params.append(status)
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    q += " ORDER BY v.scheduled_date ASC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_vaccination(vax_id: int) -> dict | None:
    conn = get_db()
    row  = conn.execute("SELECT * FROM vaccination_schedules WHERE id=?", (vax_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_vaccination(vax_id: int, data: dict) -> bool:
    conn = get_db()
    conn.execute("""
        UPDATE vaccination_schedules SET
            vaccine_name=?, scheduled_date=?, dose_number=?,
            next_due_date=?, administered_by=?, notes=?, status=?
        WHERE id=?
    """, (
        data.get('vaccine_name'), data.get('scheduled_date'),
        int(data.get('dose_number', 1)),
        data.get('next_due_date') or None,
        data.get('administered_by', ''), data.get('notes', ''),
        data.get('status', 'Pending'), vax_id
    ))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


def mark_vaccination_done(vax_id: int, administered_date: str, administered_by: str = '') -> bool:
    conn = get_db()
    conn.execute("""
        UPDATE vaccination_schedules
        SET status='Completed', administered_date=?, administered_by=?
        WHERE id=?
    """, (administered_date, administered_by, vax_id))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


def delete_vaccination(vax_id: int) -> bool:
    conn = get_db()
    conn.execute("DELETE FROM vaccination_schedules WHERE id=?", (vax_id,))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


# ══════════════════════════════════════════════════════════════════════════════
#  TREATMENTS
# ══════════════════════════════════════════════════════════════════════════════

def add_treatment(animal_uid: str, data: dict) -> int:
    conn = get_db()
    cur  = conn.execute("""
        INSERT INTO treatments
          (animal_uid, treatment_type, medicine_name, dosage,
           start_date, end_date, vet_name, cost, notes, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        animal_uid,
        data.get('treatment_type', 'Medicine'),
        data.get('medicine_name', ''),
        data.get('dosage', ''),
        data.get('start_date'),
        data.get('end_date') or None,
        data.get('vet_name', ''),
        float(data.get('cost', 0) or 0),
        data.get('notes', ''),
        _now()
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_treatments(animal_uid: str = None, status: str = None) -> list[dict]:
    conn = get_db()
    q, params = "SELECT t.*, a.name as animal_name FROM treatments t JOIN animals a ON t.animal_uid=a.uid", []
    conditions = []
    if animal_uid:
        conditions.append("t.animal_uid=?"); params.append(animal_uid)
    if status:
        conditions.append("t.status=?"); params.append(status)
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    q += " ORDER BY t.start_date DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_treatment(treat_id: int) -> dict | None:
    conn = get_db()
    row  = conn.execute("SELECT * FROM treatments WHERE id=?", (treat_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_treatment(treat_id: int, data: dict) -> bool:
    conn = get_db()
    conn.execute("""
        UPDATE treatments SET
            treatment_type=?, medicine_name=?, dosage=?,
            start_date=?, end_date=?, status=?,
            vet_name=?, cost=?, notes=?
        WHERE id=?
    """, (
        data.get('treatment_type'), data.get('medicine_name'),
        data.get('dosage'), data.get('start_date'),
        data.get('end_date') or None, data.get('status', 'Active'),
        data.get('vet_name'), float(data.get('cost', 0) or 0),
        data.get('notes'), treat_id
    ))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


def delete_treatment(treat_id: int) -> bool:
    conn = get_db()
    conn.execute("DELETE FROM treatments WHERE id=?", (treat_id,))
    affected = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    conn.close()
    return affected > 0


def get_calendar_events() -> list[dict]:
    """Return all vaccinations + treatments formatted for FullCalendar JS."""
    events = []
    STATUS_COLOUR = {
        'Pending':   '#22c55e',
        'Overdue':   '#ef4444',
        'Completed': '#64748b',
        'Skipped':   '#f97316',
        'Active':    '#60a5fa',
        'Cancelled': '#6b7280',
    }
    for v in get_vaccinations():
        events.append({
            'id':    f'vax-{v["id"]}',
            'title': f'💉 {v["vaccine_name"]} – {v["animal_name"]}',
            'start': v['scheduled_date'],
            'color': STATUS_COLOUR.get(v['status'], '#22c55e'),
            'extendedProps': {
                'type': 'vaccination', 'status': v['status'],
                'animal': v['animal_name'], 'uid': v['animal_uid'],
                'id': v['id'], 'dose': v['dose_number'],
                'notes': v['notes']
            }
        })
    for t in get_treatments():
        events.append({
            'id':    f'trx-{t["id"]}',
            'title': f'💊 {t["treatment_type"]} – {t["animal_name"]}',
            'start': t['start_date'],
            'end':   t['end_date'] or t['start_date'],
            'color': STATUS_COLOUR.get(t['status'], '#60a5fa'),
            'extendedProps': {
                'type': 'treatment', 'status': t['status'],
                'animal': t['animal_name'], 'uid': t['animal_uid'],
                'id': t['id'], 'medicine': t['medicine_name'],
                'notes': t['notes']
            }
        })
    return events
