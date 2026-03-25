# Road Watch Next Product Direction

## North Star

Road Watch should become an **AI road inspection + maintenance workflow platform**, not another pothole-reporting website.

The repo already has the inspection layer:

- multi-session video, photo, and live capture
- session scoring with `road_health_score`, `maintenance_priority`, and severity summaries
- GPS-aware session maps and hotspot generation
- CSV, PDF, KML, snapshot, and archive exports
- operator-facing dashboard and results views

That means the strongest next step is not "collect more complaints." It is closing the loop from **detection -> assignment -> repair proof -> resolution -> public trust**.

## Product Position

### Core value

Convert AI-detected road defects into maintenance work that can be assigned, tracked, verified, and handed off cleanly.

### Primary users

- operator
- reviewer
- engineer
- admin

### Anti-goal

Do not build a citizen-only intake portal unless the product also supports assignment, closure, and proof of completion. Otherwise it becomes a dead inbox.

## Why This Direction Fits The Current Repo

Road Watch already does the hard part that many civic-reporting products do not:

- it turns footage and photos into structured detection sessions
- it preserves evidence across reports, snapshots, and archives
- it already exposes maintenance-oriented summary fields
- it already has GPS-based session review and export capability

What is missing is the operational layer on top of sessions:

- who owns this issue
- what is the status
- when is it due
- what proof shows it was fixed
- how do engineers or contractors receive the packet
- how do decision-makers compare the same route over time

## Priority Roadmap

### 1. Add a repair workflow first

Convert each completed session into a work order with:

- status
- assignee
- deadline
- remarks
- priority
- linked evidence

Recommended status flow:

- `open`
- `assigned`
- `in_progress`
- `fixed`
- `verified`
- `reopened`

Why first:

- this is the shortest path from AI detection to real maintenance action
- it sits naturally on top of existing session manifests and results cards
- it prevents the product from stopping at "interesting detection demo"

### 2. Add before/after proof next

Let teams upload:

- before photos if they want extra field proof beyond snapshots
- after/fixed-road evidence
- completion remarks
- verification remarks

Then allow a pothole, hotspot, or session-linked work order to be marked resolved.

Why second:

- the repo already stores photos, snapshots, and report artifacts on disk
- this makes closure visible and auditable
- it unlocks public-facing trust pages later

### 3. Add zone and road-segment analytics

Build:

- heatmap of detections
- ward or zone summaries
- repeat-defect lists
- monthly trend charts
- unresolved-vs-resolved counts by area

Important prerequisite:

GPS alone is not enough for clean comparison. Start capturing simple route labels during upload, such as:

- `route_name`
- `road_segment`
- `ward`
- `zone`

Manual labels are acceptable first. Do not wait for full GIS map-matching to start analytics.

### 4. Add route intelligence

Compare the same road across multiple runs and show whether:

- road-health score improved or worsened
- repeat defects were reduced
- maintenance actions actually changed defect density

This becomes one of the best differentiators in the product because it measures outcomes, not just detections.

### 5. Add role-based access

Roles should be:

- `operator`: upload and inspect
- `reviewer`: validate findings, create or edit work orders
- `engineer`: receive assignments, upload fix evidence, update field remarks
- `admin`: manage users, roles, settings, and exports

Do this after work orders exist so permissions map to real actions.

### 6. Add citizen intake only after closure exists

Only add a citizen-report form when the platform can:

- assign the report
- merge it into a work order
- mark it resolved
- show closure or disposition

Otherwise skip it for now.

### 7. Add authority handoff

Provide one-click maintenance packets for engineers or contractors.

This should reuse the current archive/export idea and package:

- work-order summary
- route or zone context
- severity summary
- map snapshot or coordinates
- before evidence
- after evidence when available
- PDF or ZIP export

Later, this can extend into email-ready or WhatsApp-ready sharing.

### 8. Add a public trust layer last

Use resolved work orders to power:

- before/after gallery
- recently fixed page
- impact counters
- project stories

This should be generated from the operational system, not maintained as a separate content workflow.

## Fastest Wins For This Codebase

### A. Work-order status on top of existing sessions

This is the best first build because the backend already has:

- `session_id`
- manifest persistence
- maintenance-oriented summary fields
- recent report history

Minimal first step:

- attach a small work-order record to a session
- surface its status on the results page
- allow assignee, due date, and remarks edits

### B. Evidence upload on top of existing results

The codebase already stores snapshots, photos, annotated outputs, and archives. Add a simple evidence attachment model before building anything more complex.

### C. Heatmap and zone dashboard on top of GPS data

`/session_map/<sid>` already exposes route points, detections, bounds, and hotspot data. Use that foundation for:

- aggregate map view
- zone summary cards
- repeat hotspot ranking

### D. Login and roles on top of the current operator flow

The UI already behaves like an operator console. Add authentication after the workflow actions exist so roles protect meaningful operations.

## Suggested Phase Plan

### Phase 1: Operationalize Sessions

Goal: turn a detection session into maintainable work.

Build:

- session-to-work-order creation
- status, assignee, deadline, remarks
- evidence upload
- resolve, verify, and reopen actions

UI:

- add work-order status chips to `results.html`
- add a session detail panel or dedicated work-order detail page
- add upload controls for after-repair proof

Definition of done:

- every important session can be tracked beyond detection
- resolved work can be verified with evidence

### Phase 2: Add Area Analytics

Goal: help teams prioritize by geography and trend.

Build:

- heatmap
- zone and ward summaries
- monthly trend charts
- repeat-defect ranking

UI:

- new analytics dashboard page
- filter bar for date, zone, route, severity, and status

### Phase 3: Add Route Intelligence And Role Control

Goal: compare road condition over time and support multi-user operations.

Build:

- route or segment comparison across runs
- role-based login
- user management
- action permissions by role

UI:

- route comparison page
- login page refresh to real auth
- admin users and roles page

### Phase 4: Add Handoff And Public Trust

Goal: support external communication only after internal workflow is healthy.

Build:

- maintenance packet export
- resolved-work public gallery
- recently fixed page
- impact counters

## Recommended Storage Plan

Do not rewrite the app yet.

Recommended approach:

- keep media and generated artifacts on disk as the repo already does
- add a lightweight metadata database first
- use SQLite in the current Flask setup
- move to Postgres only when the app becomes multi-user beyond local or small-team usage

Recommended tables:

### `users`

- `id`
- `name`
- `email`
- `password_hash`
- `role`
- `active`
- `created_at`

### `sessions`

Use this as a metadata mirror for important session fields already produced by the backend.

- `session_id`
- `video_name`
- `source_type`
- `created_at`
- `road_health_score`
- `maintenance_priority`
- `zone`
- `ward`
- `route_name`
- `road_segment`
- `status`

### `work_orders`

- `id`
- `session_id`
- `title`
- `status`
- `priority`
- `assignee_user_id`
- `deadline`
- `remarks`
- `resolved_at`
- `verified_at`
- `created_by`
- `created_at`
- `updated_at`

### `work_order_events`

Track status changes and remarks over time.

- `id`
- `work_order_id`
- `event_type`
- `old_value`
- `new_value`
- `remarks`
- `created_by`
- `created_at`

### `evidence_files`

- `id`
- `work_order_id`
- `session_id`
- `kind`
- `file_path`
- `caption`
- `uploaded_by`
- `created_at`

Suggested `kind` values:

- `before`
- `after`
- `verification`
- `supporting`

### `route_runs`

Use this once route comparison begins.

- `id`
- `session_id`
- `route_name`
- `road_segment`
- `zone`
- `ward`
- `run_date`
- `road_health_score`
- `total_detections`
- `total_area`

## API Changes To Add

Keep existing inspection endpoints intact. Layer workflow endpoints beside them.

Suggested first additions:

- `POST /sessions/<sid>/work_order`
- `GET /work_orders`
- `GET /work_orders/<id>`
- `PATCH /work_orders/<id>`
- `POST /work_orders/<id>/evidence`
- `POST /work_orders/<id>/resolve`
- `POST /work_orders/<id>/verify`
- `POST /work_orders/<id>/reopen`
- `GET /analytics/zones`
- `GET /analytics/routes`
- `GET /routes/<route_name>/history`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`

Keep using current endpoints as source data:

- `/upload`
- `/upload_photo`
- `/live/start`
- `/session_state/<sid>`
- `/recent_reports`
- `/session_map/<sid>`

## UI Pages To Add

Recommended order:

### 1. Extend the existing results page

Add:

- work-order status chip
- assignee and deadline display
- create work-order button
- evidence upload action
- resolve and verify controls

### 2. Add a work-order detail page

Show:

- session summary
- linked artifacts
- status timeline
- before and after gallery
- remarks and verification notes

### 3. Add an analytics page

Show:

- map heat layer
- zone summaries
- monthly trends
- repeat-defect list
- unresolved queue by area

### 4. Add a route comparison page

Show:

- same-road history
- score deltas
- defect trend over time
- latest maintenance actions

### 5. Add login and admin pages

Show:

- user login
- user list
- role management
- permission-scoped actions

### 6. Add public proof pages later

Show:

- recently fixed roads
- before/after evidence
- impact counters

## What To Build First In This Flask + HTML Setup

If only one path is followed, build in this order:

1. Add work-order metadata to sessions and expose it in the results UI.
2. Add evidence upload and resolution workflow on top of that.
3. Add aggregate map and zone analytics using existing GPS data.
4. Add route labels and historical route comparison.
5. Add auth and role-based control once those actions exist.
6. Add citizen intake and public-facing trust pages only after closure is reliable.

## Practical Architecture Notes

- Do not replace Flask just to add workflow features.
- Do not wait for a full SPA rewrite before adding operations.
- Do not start with citizen reporting.
- Use the current archive export as the base for the future maintenance packet.
- Capture simple route metadata during upload before attempting advanced route intelligence.

## Reference Products And Inspiration

These are useful references, but Road Watch should not copy them directly. The better opportunity is to combine AI inspection with maintenance closure.

- https://www.indianpotholes.com/
- https://github.com/Empowered-Indian/indian-potholes/blob/master/frontend/README.md
- https://github.com/Empowered-Indian/indian-potholes
- https://www.potholeraja.com/
- https://nammapothole.com/
