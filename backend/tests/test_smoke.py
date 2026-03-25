import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


TEMP_ROOT = tempfile.TemporaryDirectory()
TEMP_PATH = Path(TEMP_ROOT.name)
os.environ.setdefault("MODEL_PATH", str(TEMP_PATH / "missing-model.pt"))
os.environ.setdefault("UPLOAD_DIR", str(TEMP_PATH / "uploads"))
os.environ.setdefault("REPORT_DIR", str(TEMP_PATH / "reports"))
os.environ.setdefault("WORKFLOW_DB_PATH", str(TEMP_PATH / "workflow.db"))

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as road_watch_app


class WorkflowSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = road_watch_app.app.test_client()

    def setUp(self):
        self.client.delete("/sessions")

    def seed_session(self, video_name="phase1-demo.mp4"):
        entry = road_watch_app.create_session_entry("", video_name, source_type="video")
        sid = entry["session_id"]
        with road_watch_app.sessions_lock:
            session = road_watch_app.sessions[sid]
            session["total_detections"] = 4
            session["total_area"] = 1.37
            session["average_confidence"] = 0.912
            session["severity_counts"] = {
                "Minor": 1,
                "Moderate": 2,
                "Major": 1,
                "Severe": 0,
            }
            session["road_health_score"] = 61
            session["health_band"] = "Fair"
            session["maintenance_priority"] = "High"
            session["insight_headline"] = "Surface degradation detected."
            session["recommended_action"] = "Dispatch a repair crew."
        manifest_path = road_watch_app.save_session_manifest(sid)
        self.assertTrue(manifest_path)
        self.assertTrue(os.path.exists(manifest_path))
        return sid

    def upload_evidence(self, work_order_id, kind, caption):
        response = self.client.post(
            f"/work_orders/{work_order_id}/evidence",
            data={
                "kind": kind,
                "caption": caption,
                "uploaded_by": "Engineer A",
                "file": (io.BytesIO(b"fake-image-bytes"), f"{kind}.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.get_json()

    def test_phase_one_workflow_round_trip(self):
        session_id = self.seed_session()

        create_response = self.client.post(
            f"/sessions/{session_id}/work_order",
            json={
                "title": "Repair potholes near sector 8",
                "status": "assigned",
                "priority": "High",
                "assignee": "Engineer A",
                "deadline": "2026-03-31",
                "remarks": "Initial maintenance order created from AI scan.",
                "actor": "Operator 1",
            },
        )
        self.assertEqual(create_response.status_code, 201, create_response.get_data(as_text=True))
        create_data = create_response.get_json()
        work_order = create_data["work_order"]
        work_order_id = work_order["id"]
        self.assertEqual(work_order["status"], "assigned")
        self.assertEqual(work_order["assignee"], "Engineer A")
        self.assertEqual(work_order["deadline"], "2026-03-31")

        get_session_response = self.client.get(f"/sessions/{session_id}/work_order")
        self.assertEqual(get_session_response.status_code, 200)
        self.assertTrue(get_session_response.get_json()["has_work_order"])

        patch_response = self.client.patch(
            f"/work_orders/{work_order_id}",
            json={
                "status": "in_progress",
                "assignee": "Crew Bravo",
                "remarks": "Crew dispatched and lane closure arranged.",
                "actor": "Reviewer 1",
            },
        )
        self.assertEqual(patch_response.status_code, 200, patch_response.get_data(as_text=True))
        patched = patch_response.get_json()["work_order"]
        self.assertEqual(patched["status"], "in_progress")
        self.assertEqual(patched["assignee"], "Crew Bravo")

        evidence_after = self.upload_evidence(work_order_id, "after", "Patched surface after repair.")
        self.assertTrue(evidence_after["work_order"]["has_after_proof"])

        evidence_verify = self.upload_evidence(work_order_id, "verification", "Reviewer closeout photo.")
        self.assertTrue(evidence_verify["work_order"]["has_verification_proof"])

        resolve_response = self.client.post(
            f"/work_orders/{work_order_id}/resolve",
            json={"remarks": "Repair completed and compacted.", "actor": "Engineer A"},
        )
        self.assertEqual(resolve_response.status_code, 200, resolve_response.get_data(as_text=True))
        resolved = resolve_response.get_json()["work_order"]
        self.assertEqual(resolved["status"], "fixed")
        self.assertIsNotNone(resolved["resolved_at"])

        verify_response = self.client.post(
            f"/work_orders/{work_order_id}/verify",
            json={"remarks": "Verified closed after site visit.", "actor": "Reviewer 2"},
        )
        self.assertEqual(verify_response.status_code, 200, verify_response.get_data(as_text=True))
        verified = verify_response.get_json()["work_order"]
        self.assertEqual(verified["status"], "verified")
        self.assertIsNotNone(verified["verified_at"])
        self.assertEqual(verified["evidence_count"], 2)

        detail_response = self.client.get(f"/work_orders/{work_order_id}")
        self.assertEqual(detail_response.status_code, 200, detail_response.get_data(as_text=True))
        detail_data = detail_response.get_json()["work_order"]
        self.assertEqual(len(detail_data["evidence_files"]), 2)
        self.assertGreaterEqual(len(detail_data["events"]), 6)

        recent_response = self.client.get("/recent_reports?limit=10")
        self.assertEqual(recent_response.status_code, 200)
        items = recent_response.get_json()["items"]
        matching = [item for item in items if item["session_id"] == session_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["work_order"]["status"], "verified")

        evidence_path = detail_data["evidence_files"][0]["file_path"]
        self.assertTrue(os.path.exists(evidence_path))

        delete_response = self.client.delete(f"/sessions/{session_id}")
        self.assertEqual(delete_response.status_code, 200, delete_response.get_data(as_text=True))
        self.assertFalse(os.path.exists(evidence_path))

        missing_work_order = self.client.get(f"/work_orders/{work_order_id}")
        self.assertEqual(missing_work_order.status_code, 404)


if __name__ == "__main__":
    unittest.main()
