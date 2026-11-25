import uuid
import httpx
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from app.config import settings
from app.storage.database import DatabaseManager, get_db
from app.storage.models import Lead, Appointment, CRMRecord


class CRMTool:
    """
    CRM Integration Tool.
    Synchronizes qualified leads, BANT scores, call history, and appointment details
    into CRM systems (Mock CRM, REST Webhooks, or HubSpot).
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db()
        self.mode = settings.CRM_MODE
        self.webhook_url = settings.CRM_WEBHOOK_URL

    def sync_lead(
        self,
        lead: Lead,
        appointment: Optional[Appointment] = None,
        notes: Optional[str] = None
    ) -> Tuple[bool, str, CRMRecord]:
        """
        Synchronizes a qualified lead and any associated appointment to the CRM.
        Returns: (success: bool, message: str, record: CRMRecord)
        """
        if not lead.id:
            lead = self.db.save_lead(lead)

        payload: Dict[str, Any] = {
            "lead_id": lead.id,
            "session_id": lead.session_id,
            "name": lead.name or "Unknown Prospect",
            "email": lead.email or "unspecified@prospect.com",
            "company": lead.company or "Unspecified",
            "role": lead.role or "Unspecified",
            "budget": lead.budget,
            "authority": lead.authority,
            "need": lead.need,
            "timeline": lead.timeline,
            "lead_score": lead.lead_score,
            "qualification_status": lead.qualification_status.value,
            "synced_at": datetime.now(timezone.utc).isoformat()
        }

        if appointment:
            payload["appointment"] = {
                "appointment_id": appointment.id,
                "title": appointment.title,
                "start_time": appointment.start_time,
                "end_time": appointment.end_time,
                "status": appointment.status.value,
                "notes": appointment.notes
            }

        if notes:
            payload["agent_notes"] = notes

        # Mode: Webhook or HubSpot integration
        if self.mode in ["webhook", "hubspot"] and self.webhook_url:
            try:
                response = httpx.post(self.webhook_url, json=payload, timeout=5.0)
                if response.status_code in [200, 201, 202]:
                    external_id = response.json().get("id", f"WEBHOOK-{uuid.uuid4().hex[:8]}")
                    record = CRMRecord(
                        lead_id=lead.id,
                        crm_system=self.mode,
                        external_id=external_id,
                        sync_status="SYNCED",
                        payload=payload
                    )
                    saved = self.db.save_crm_record(record)
                    return True, f"Successfully synced lead to external CRM ({self.mode}).", saved
                else:
                    record = CRMRecord(
                        lead_id=lead.id,
                        crm_system=self.mode,
                        external_id=None,
                        sync_status="FAILED",
                        payload={"error": f"HTTP {response.status_code}", "raw_payload": payload}
                    )
                    saved = self.db.save_crm_record(record)
                    return False, f"External CRM returned status code {response.status_code}.", saved
            except Exception as e:
                record = CRMRecord(
                    lead_id=lead.id,
                    crm_system=self.mode,
                    external_id=None,
                    sync_status="FAILED",
                    payload={"error": str(e), "raw_payload": payload}
                )
                saved = self.db.save_crm_record(record)
                return False, f"Failed to connect to external CRM: {str(e)}", saved

        # Default: High-fidelity Mock CRM
        external_id = f"MOCK-CRM-{uuid.uuid4().hex[:8].upper()}"
        record = CRMRecord(
            lead_id=lead.id,
            crm_system="mock",
            external_id=external_id,
            sync_status="SYNCED",
            payload=payload
        )
        saved = self.db.save_crm_record(record)
        return True, f"Lead successfully synced to Mock CRM with ID: {external_id}", saved
