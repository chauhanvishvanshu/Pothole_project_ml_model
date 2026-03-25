import csv
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime
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

    def write_csv_logs(self, session_id, logs):
        csv_path = Path(os.environ["REPORT_DIR"]) / f"{session_id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame_number",
                    "video_time_s",
                    "timestamp",
                    "confidence",
                    "area_m2",
                    "severity",
                    "track_id",
                    "snapshot_file",
                    "latitude",
                    "longitude",
                ],
            )
            writer.writeheader()
            for index, log in enumerate(logs, start=1):
                writer.writerow({
                    "frame_number": index,
                    "video_time_s": log.get("video_time_s", float(index)),
                    "timestamp": log.get("timestamp", f"2026-01-01T00:00:{index:02d}Z"),
                    "confidence": log.get("confidence", 0.8),
                    "area_m2": log.get("area_m2", 0.5),
                    "severity": log.get("severity", "Minor"),
                    "track_id": log.get("track_id", index),
                    "snapshot_file": log.get("snapshot_file", ""),
                    "latitude": log.get("latitude", ""),
                    "longitude": log.get("longitude", ""),
                })
        return str(csv_path)

    def stamp_manifest_time(self, session_id, timestamp_value):
        manifest = road_watch_app.load_manifest_record(session_id)
        self.assertIsNotNone(manifest)
        manifest["_manifest_path"] = manifest.get("_manifest_path")
        manifest["generated_at"] = timestamp_value
        manifest["created_at"] = timestamp_value
        manifest["last_activity"] = timestamp_value
        road_watch_app.write_manifest_record(manifest["_manifest_path"], manifest)
        os.utime(manifest["_manifest_path"], (timestamp_value, timestamp_value))

    def seed_session(self, video_name="phase1-demo.mp4", metadata=None, logs=None, generated_at=None):
        entry = road_watch_app.create_session_entry("", video_name, source_type="video", metadata=metadata)
        sid = entry["session_id"]
        logs = logs or []
        summary = road_watch_app.summarize_logs(logs)
        csv_path = self.write_csv_logs(sid, logs) if logs else None
        with road_watch_app.sessions_lock:
            session = road_watch_app.sessions[sid]
            session["detection_logs"] = list(logs)
            session["csv_path"] = csv_path
            if csv_path:
                session["csv_url"] = f"/reports/{Path(csv_path).name}"
            session["total_detections"] = summary["detections"] if logs else 4
            session["total_area"] = summary["total_area"] if logs else 1.37
            session["average_confidence"] = summary["avg_confidence"] if logs else 0.912
            session["severity_counts"] = summary["severity_counts"] if logs else {
                "Minor": 1,
                "Moderate": 2,
                "Major": 1,
                "Severe": 0,
            }
            session["unique_hazards"] = summary["unique_hazards"] if logs else 3
            session["top_severity"] = summary["top_severity"] if logs else "Major"
            session["road_health_score"] = summary["road_health_score"] if logs else 61
            session["health_band"] = summary["health_band"] if logs else "Fair"
            session["maintenance_priority"] = summary["maintenance_priority"] if logs else "High"
            session["insight_headline"] = summary["insight_headline"] if logs else "Surface degradation detected."
            session["recommended_action"] = summary["recommended_action"] if logs else "Dispatch a repair crew."
            session["gps_detection_count"] = summary["gps_detection_count"] if logs else 0
            session["has_map"] = summary["has_map"] if logs else False
            session["hotspot_count"] = summary["hotspot_count"] if logs else 0
            session["hotspots"] = summary["hotspots"] if logs else []
            session["highlights"] = summary["highlights"] if logs else []
            if generated_at:
                session["created_at"] = generated_at
                session["last_activity"] = generated_at
        manifest_path = road_watch_app.save_session_manifest(sid)
        self.assertTrue(manifest_path)
        self.assertTrue(os.path.exists(manifest_path))
        if generated_at:
            self.stamp_manifest_time(sid, generated_at)
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

    def test_phase_two_analytics_summary_and_filters(self):
        alpha_session = self.seed_session(
            video_name="north-alpha-jan.mp4",
            metadata={
                "zone": "North Zone",
                "ward": "Ward 11",
                "route_name": "Alpha Corridor",
                "road_segment": "Segment A",
            },
            logs=[
                {"video_time_s": 3.5, "confidence": 0.92, "area_m2": 0.85, "severity": "Moderate", "latitude": 12.9716, "longitude": 77.5946},
                {"video_time_s": 8.2, "confidence": 0.88, "area_m2": 1.15, "severity": "Major", "latitude": 12.9718, "longitude": 77.5948},
            ],
            generated_at=datetime(2026, 1, 10, 9, 30, 0).timestamp(),
        )
        beta_session = self.seed_session(
            video_name="north-alpha-feb.mp4",
            metadata={
                "zone": "North Zone",
                "ward": "Ward 12",
                "route_name": "Alpha Corridor",
                "road_segment": "Segment A",
            },
            logs=[
                {"video_time_s": 2.3, "confidence": 0.96, "area_m2": 1.6, "severity": "Severe", "latitude": 12.9717, "longitude": 77.5947},
                {"video_time_s": 5.6, "confidence": 0.84, "area_m2": 1.05, "severity": "Major", "latitude": 12.9717, "longitude": 77.5947},
                {"video_time_s": 9.1, "confidence": 0.91, "area_m2": 1.3, "severity": "Severe", "latitude": 12.9719, "longitude": 77.5950},
            ],
            generated_at=datetime(2026, 2, 12, 10, 15, 0).timestamp(),
        )
        gamma_session = self.seed_session(
            video_name="south-beta-feb.mp4",
            metadata={
                "zone": "South Zone",
                "ward": "Ward 7",
                "route_name": "Beta Link",
                "road_segment": "Segment B",
            },
            logs=[
                {"video_time_s": 4.8, "confidence": 0.78, "area_m2": 0.45, "severity": "Minor", "latitude": 12.9815, "longitude": 77.6101},
            ],
            generated_at=datetime(2026, 2, 20, 18, 45, 0).timestamp(),
        )

        for session_id, status, assignee in (
            (alpha_session, "assigned", "Crew Alpha"),
            (beta_session, "verified", "Crew Bravo"),
            (gamma_session, "fixed", "Crew Charlie"),
        ):
            response = self.client.post(
                f"/sessions/{session_id}/work_order",
                json={
                    "title": f"Work order for {session_id}",
                    "status": status,
                    "priority": "High",
                    "assignee": assignee,
                    "deadline": "2026-03-31",
                    "remarks": "Analytics smoke test seed.",
                    "actor": "Test Operator",
                },
            )
            self.assertIn(response.status_code, (200, 201), response.get_data(as_text=True))

        overview_response = self.client.get("/analytics/overview")
        self.assertEqual(overview_response.status_code, 200, overview_response.get_data(as_text=True))
        overview = overview_response.get_json()
        self.assertEqual(overview["summary"]["session_count"], 3)
        self.assertEqual(overview["summary"]["detection_count"], 6)
        self.assertEqual(overview["summary"]["resolved_count"], 2)
        self.assertEqual(overview["summary"]["verified_count"], 1)
        self.assertEqual(overview["summary"]["open_work_orders"], 1)
        self.assertEqual(overview["summary"]["zone_count"], 2)
        self.assertEqual(overview["summary"]["route_count"], 2)
        self.assertEqual(len(overview["zone_summaries"]), 2)
        self.assertEqual(len(overview["route_summaries"]), 2)
        self.assertEqual(overview["monthly_trends"][0]["month"], "2026-01")
        self.assertEqual(overview["monthly_trends"][1]["month"], "2026-02")
        self.assertIn("North Zone", overview["filters"]["zones"])
        self.assertIn("South Zone", overview["filters"]["zones"])

        zones_response = self.client.get("/analytics/zones")
        self.assertEqual(zones_response.status_code, 200, zones_response.get_data(as_text=True))
        zones = zones_response.get_json()
        self.assertGreaterEqual(len(zones["heat_points"]), 3)
        self.assertEqual(zones["zone_summaries"][0]["zone"], "North Zone")

        routes_response = self.client.get("/analytics/routes")
        self.assertEqual(routes_response.status_code, 200, routes_response.get_data(as_text=True))
        routes = routes_response.get_json()
        self.assertEqual(len(routes["route_summaries"]), 2)
        self.assertEqual(routes["route_summaries"][0]["route_name"], "Alpha Corridor")
        self.assertEqual(routes["monthly_trends"][0]["month"], "2026-01")
        self.assertEqual(routes["monthly_trends"][1]["month"], "2026-02")

        north_only = self.client.get("/analytics/overview?zone=North%20Zone")
        self.assertEqual(north_only.status_code, 200)
        north_summary = north_only.get_json()["summary"]
        self.assertEqual(north_summary["session_count"], 2)
        self.assertEqual(north_summary["route_count"], 1)

        severe_only = self.client.get("/analytics/zones?severity=Severe")
        self.assertEqual(severe_only.status_code, 200)
        severe_payload = severe_only.get_json()
        self.assertEqual(severe_payload["summary"]["session_count"], 1)
        self.assertTrue(severe_payload["heat_points"])
        self.assertTrue(all(point["severity"] == "Severe" for point in severe_payload["heat_points"]))

        assigned_only = self.client.get("/analytics/routes?status=assigned")
        self.assertEqual(assigned_only.status_code, 200)
        assigned_payload = assigned_only.get_json()
        self.assertEqual(assigned_payload["summary"]["session_count"], 1)
        self.assertEqual(len(assigned_payload["route_summaries"]), 1)
        self.assertEqual(assigned_payload["route_summaries"][0]["route_name"], "Alpha Corridor")


if __name__ == "__main__":
    unittest.main()
