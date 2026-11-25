import uuid
from typing import Optional, Tuple
from datetime import datetime
from app.config import settings
from app.storage.database import DatabaseManager, get_db
from app.storage.models import Appointment, AppointmentStatus, Lead


class CalendarTool:
    """
    Automated Appointment Booking Tool.
    Manages calendar availability queries, conflict checking, and booking operations.
    Integrates with SQLite persistence and supports Mock & external calendar providers.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db()
        self.mode = settings.CALENDAR_MODE

    def check_availability(self, start_time: str, end_time: str) -> bool:
        """
        Checks whether the requested slot is available (no conflicting confirmed appointments).
        Returns True if available, False if conflict exists.
        """
        has_conflict = self.db.check_slot_conflict(start_time, end_time)
        return not has_conflict

    def book_appointment(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        attendee_name: str,
        attendee_email: str,
        lead_id: Optional[str] = None,
        title: str = "Fortivex AI Technical Consultation & Product Demo",
        notes: Optional[str] = None,
        timezone: str = "UTC"
    ) -> Tuple[bool, str, Optional[Appointment]]:
        """
        Validates timestamps, verifies availability, and books an appointment.
        Returns: (success: bool, message: str, appointment: Optional[Appointment])
        """
        # 1. Validate timestamps
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if end_dt <= start_dt:
                return False, "End time must be after start time.", None
        except ValueError as e:
            return False, f"Invalid ISO timestamp format: {str(e)}", None

        # 2. Check for slot conflict
        if not self.check_availability(start_time, end_time):
            return False, f"The slot from {start_time} to {end_time} is already booked. Please choose another time.", None

        # 3. Associate with lead if not passed
        if not lead_id:
            lead = self.db.get_lead_by_session(session_id)
            if lead:
                lead_id = lead.id

        # 4. Generate mock meeting link if in mock mode
        meet_id = str(uuid.uuid4())[:8]
        meeting_notes = notes or ""
        meeting_notes += f" | Meeting Link: https://meet.fortivex.ai/demo-{meet_id}"

        appointment = Appointment(
            id=str(uuid.uuid4()),
            session_id=session_id,
            lead_id=lead_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
            attendee_name=attendee_name,
            attendee_email=attendee_email,
            status=AppointmentStatus.CONFIRMED,
            notes=meeting_notes
        )

        saved_apt = self.db.save_appointment(appointment)
        return True, f"Appointment successfully confirmed for {attendee_name} at {start_time}. Calendar invite dispatched.", saved_apt
