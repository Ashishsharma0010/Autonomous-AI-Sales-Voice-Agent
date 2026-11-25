import json
import sqlite3
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from app.storage.models import (
    Lead, Appointment, CallLog, CRMRecord,
    LeadStatus, AppointmentStatus, CallTurn
)
from app.config import settings


class DatabaseManager:
    """
    SQLite Database Manager for persistence of Leads, Appointments, Call Logs, and CRM Records.
    Thread-safe synchronous implementation for guaranteed consistency and zero-config deployment.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.DATABASE_PATH

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initializes database schema if tables do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Leads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    email TEXT,
                    company TEXT,
                    role TEXT,
                    budget TEXT,
                    authority TEXT,
                    need TEXT,
                    timeline TEXT,
                    lead_score INTEGER DEFAULT 0,
                    qualification_status TEXT DEFAULT 'UNQUALIFIED',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # Appointments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    lead_id TEXT,
                    title TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    timezone TEXT DEFAULT 'UTC',
                    attendee_name TEXT,
                    attendee_email TEXT,
                    status TEXT DEFAULT 'CONFIRMED',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE SET NULL
                );
            """)

            # Call logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    lead_id TEXT,
                    transcript_json TEXT NOT NULL,
                    summary TEXT,
                    key_takeaways_json TEXT,
                    action_items_json TEXT,
                    sentiment_trajectory_json TEXT,
                    detected_objections_json TEXT,
                    sales_insights_json TEXT,
                    duration_seconds INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE SET NULL
                );
            """)

            # CRM records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crm_records (
                    id TEXT PRIMARY KEY,
                    lead_id TEXT NOT NULL,
                    crm_system TEXT DEFAULT 'mock',
                    external_id TEXT,
                    sync_status TEXT DEFAULT 'SYNCED',
                    payload_json TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE CASCADE
                );
            """)

            conn.commit()

    # ==================== LEAD METHODS ====================

    def save_lead(self, lead: Lead) -> Lead:
        """Creates or updates a lead based on session_id or id."""
        if not lead.id:
            lead.id = str(uuid.uuid4())
        lead.updated_at = datetime.now(timezone.utc).isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO leads (
                    id, session_id, name, email, company, role, budget, authority,
                    need, timeline, lead_score, qualification_status, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    name = COALESCE(excluded.name, leads.name),
                    email = COALESCE(excluded.email, leads.email),
                    company = COALESCE(excluded.company, leads.company),
                    role = COALESCE(excluded.role, leads.role),
                    budget = COALESCE(excluded.budget, leads.budget),
                    authority = COALESCE(excluded.authority, leads.authority),
                    need = COALESCE(excluded.need, leads.need),
                    timeline = COALESCE(excluded.timeline, leads.timeline),
                    lead_score = excluded.lead_score,
                    qualification_status = excluded.qualification_status,
                    notes = COALESCE(excluded.notes, leads.notes),
                    updated_at = excluded.updated_at;
            """, (
                lead.id, lead.session_id, lead.name, lead.email, lead.company,
                lead.role, lead.budget, lead.authority, lead.need, lead.timeline,
                lead.lead_score, lead.qualification_status.value, lead.notes,
                lead.created_at, lead.updated_at
            ))
            conn.commit()
            return self.get_lead_by_session(lead.session_id) or lead

    def get_lead(self, lead_id: str) -> Optional[Lead]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            return self._row_to_lead(row) if row else None

    def get_lead_by_session(self, session_id: str) -> Optional[Lead]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM leads WHERE session_id = ?", (session_id,)).fetchone()
            return self._row_to_lead(row) if row else None

    def list_leads(self) -> List[Lead]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM leads ORDER BY created_at DESC").fetchall()
            return [self._row_to_lead(row) for row in rows]

    # ==================== APPOINTMENT METHODS ====================

    def check_slot_conflict(self, start_time: str, end_time: str, exclude_id: Optional[str] = None) -> bool:
        """Returns True if there is an overlapping confirmed appointment."""
        with self.get_connection() as conn:
            query = """
                SELECT COUNT(*) as count FROM appointments
                WHERE status = 'CONFIRMED'
                AND ((start_time < ? AND end_time > ?) OR (start_time >= ? AND start_time < ?))
            """
            params = [end_time, start_time, start_time, end_time]
            if exclude_id:
                query += " AND id != ?"
                params.append(exclude_id)
            row = conn.execute(query, tuple(params)).fetchone()
            return row["count"] > 0 if row else False

    def save_appointment(self, appointment: Appointment) -> Appointment:
        if not appointment.id:
            appointment.id = str(uuid.uuid4())

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO appointments (
                    id, session_id, lead_id, title, start_time, end_time,
                    timezone, attendee_name, attendee_email, status, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    notes = excluded.notes;
            """, (
                appointment.id, appointment.session_id, appointment.lead_id,
                appointment.title, appointment.start_time, appointment.end_time,
                appointment.timezone, appointment.attendee_name,
                appointment.attendee_email, appointment.status.value,
                appointment.notes, appointment.created_at
            ))
            conn.commit()
            return appointment

    def get_appointments_by_lead(self, lead_id: str) -> List[Appointment]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM appointments WHERE lead_id = ? ORDER BY start_time ASC", (lead_id,)).fetchall()
            return [self._row_to_appointment(row) for row in rows]

    def list_appointments(self) -> List[Appointment]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM appointments ORDER BY start_time ASC").fetchall()
            return [self._row_to_appointment(row) for row in rows]

    # ==================== CALL LOG METHODS ====================

    def save_call_log(self, call_log: CallLog) -> CallLog:
        if not call_log.id:
            call_log.id = str(uuid.uuid4())

        transcript_data = [t.model_dump() for t in call_log.transcript]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO call_logs (
                    id, session_id, lead_id, transcript_json, summary,
                    key_takeaways_json, action_items_json, sentiment_trajectory_json,
                    detected_objections_json, sales_insights_json, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    transcript_json = excluded.transcript_json,
                    summary = excluded.summary,
                    key_takeaways_json = excluded.key_takeaways_json,
                    action_items_json = excluded.action_items_json,
                    sentiment_trajectory_json = excluded.sentiment_trajectory_json,
                    detected_objections_json = excluded.detected_objections_json,
                    sales_insights_json = excluded.sales_insights_json,
                    duration_seconds = excluded.duration_seconds;
            """, (
                call_log.id, call_log.session_id, call_log.lead_id,
                json.dumps(transcript_data), call_log.summary,
                json.dumps(call_log.key_takeaways),
                json.dumps(call_log.action_items),
                json.dumps(call_log.sentiment_trajectory),
                json.dumps(call_log.detected_objections),
                json.dumps(call_log.sales_insights),
                call_log.duration_seconds, call_log.created_at
            ))
            conn.commit()
            return call_log

    def get_call_log_by_session(self, session_id: str) -> Optional[CallLog]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM call_logs WHERE session_id = ?", (session_id,)).fetchone()
            return self._row_to_call_log(row) if row else None

    # ==================== CRM RECORD METHODS ====================

    def save_crm_record(self, record: CRMRecord) -> CRMRecord:
        if not record.id:
            record.id = str(uuid.uuid4())

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO crm_records (
                    id, lead_id, crm_system, external_id, sync_status, payload_json, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.lead_id, record.crm_system, record.external_id,
                record.sync_status, json.dumps(record.payload), record.synced_at
            ))
            conn.commit()
            return record

    def list_crm_records(self, lead_id: Optional[str] = None) -> List[CRMRecord]:
        with self.get_connection() as conn:
            if lead_id:
                rows = conn.execute("SELECT * FROM crm_records WHERE lead_id = ? ORDER BY synced_at DESC", (lead_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM crm_records ORDER BY synced_at DESC").fetchall()
            return [self._row_to_crm_record(row) for row in rows]

    # ==================== HELPERS ====================

    @staticmethod
    def _row_to_lead(row: sqlite3.Row) -> Lead:
        return Lead(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            email=row["email"],
            company=row["company"],
            role=row["role"],
            budget=row["budget"],
            authority=row["authority"],
            need=row["need"],
            timeline=row["timeline"],
            lead_score=row["lead_score"],
            qualification_status=LeadStatus(row["qualification_status"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    @staticmethod
    def _row_to_appointment(row: sqlite3.Row) -> Appointment:
        return Appointment(
            id=row["id"],
            session_id=row["session_id"],
            lead_id=row["lead_id"],
            title=row["title"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            timezone=row["timezone"],
            attendee_name=row["attendee_name"],
            attendee_email=row["attendee_email"],
            status=AppointmentStatus(row["status"]),
            notes=row["notes"],
            created_at=row["created_at"]
        )

    @staticmethod
    def _row_to_call_log(row: sqlite3.Row) -> CallLog:
        raw_turns = json.loads(row["transcript_json"]) if row["transcript_json"] else []
        turns = [CallTurn(**t) for t in raw_turns]
        return CallLog(
            id=row["id"],
            session_id=row["session_id"],
            lead_id=row["lead_id"],
            transcript=turns,
            summary=row["summary"],
            key_takeaways=json.loads(row["key_takeaways_json"]) if row["key_takeaways_json"] else [],
            action_items=json.loads(row["action_items_json"]) if row["action_items_json"] else [],
            sentiment_trajectory=json.loads(row["sentiment_trajectory_json"]) if row["sentiment_trajectory_json"] else [],
            detected_objections=json.loads(row["detected_objections_json"]) if row["detected_objections_json"] else [],
            sales_insights=json.loads(row["sales_insights_json"]) if row["sales_insights_json"] else {},
            duration_seconds=row["duration_seconds"],
            created_at=row["created_at"]
        )

    @staticmethod
    def _row_to_crm_record(row: sqlite3.Row) -> CRMRecord:
        return CRMRecord(
            id=row["id"],
            lead_id=row["lead_id"],
            crm_system=row["crm_system"],
            external_id=row["external_id"],
            sync_status=row["sync_status"],
            payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
            synced_at=row["synced_at"]
        )


# Singleton factory
_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: Optional[str] = None) -> DatabaseManager:
    global _db_instance
    if _db_instance is None or db_path is not None:
        _db_instance = DatabaseManager(db_path)
        _db_instance.init_db()
    return _db_instance
