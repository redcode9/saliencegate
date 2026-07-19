PRAGMA application_id = 0x53474350;

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version >= 1),
    name TEXT NOT NULL UNIQUE CHECK (
        typeof(name) = 'text' AND length(name) BETWEEN 1 AND 100
    ),
    checksum TEXT NOT NULL CHECK (
        typeof(checksum) = 'text'
        AND length(checksum) = 64
        AND checksum NOT GLOB '*[^0-9a-f]*'
    )
) WITHOUT ROWID;

CREATE TABLE connections (
    connection_id TEXT PRIMARY KEY CHECK (
        typeof(connection_id) = 'text' AND length(connection_id) BETWEEN 1 AND 256
    ),
    project_digest TEXT NOT NULL CHECK (
        typeof(project_digest) = 'text'
        AND length(project_digest) = 64
        AND project_digest NOT GLOB '*[^0-9a-f]*'
    ),
    profile_id TEXT NOT NULL CHECK (
        typeof(profile_id) = 'text'
        AND profile_id IN (
            'codex-hooks/v1',
            'claude-code-hooks/v1',
            'opencode-plugin/v1',
            'pi-extension/v1'
        )
    ),
    capability_manifest_digest TEXT NOT NULL CHECK (
        typeof(capability_manifest_digest) = 'text'
        AND length(capability_manifest_digest) = 64
        AND capability_manifest_digest NOT GLOB '*[^0-9a-f]*'
    ),
    host_version TEXT NOT NULL CHECK (
        typeof(host_version) = 'text' AND length(host_version) BETWEEN 1 AND 64
    ),
    compatibility_status TEXT NOT NULL CHECK (
        typeof(compatibility_status) = 'text'
        AND compatibility_status IN (
            'verified',
            'schema_compatible_unverified_version'
        )
    ),
    state TEXT NOT NULL CHECK (
        typeof(state) = 'text'
        AND state IN ('pending', 'enabled', 'draining', 'disabled', 'deleting')
    ),
    created_at TEXT NOT NULL CHECK (
        typeof(created_at) = 'text' AND length(created_at) BETWEEN 20 AND 40
    ),
    updated_at TEXT NOT NULL CHECK (
        typeof(updated_at) = 'text' AND length(updated_at) BETWEEN 20 AND 40
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    )
) WITHOUT ROWID;

CREATE INDEX connections_project_idx ON connections(project_digest, connection_id);

CREATE TABLE capture_sessions (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL CHECK (
        typeof(session_id) = 'text'
        AND length(session_id) = 64
        AND session_id NOT GLOB '*[^0-9a-f]*'
    ),
    human_id TEXT NOT NULL UNIQUE CHECK (
        typeof(human_id) = 'text'
        AND length(human_id) BETWEEN 12 AND 52
        AND human_id NOT GLOB '*[^a-z2-7]*'
    ),
    state TEXT NOT NULL CHECK (
        typeof(state) = 'text'
        AND state IN ('open', 'closed', 'quarantined', 'deleting')
    ),
    event_count INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(event_count) = 'integer' AND event_count BETWEEN 0 AND 1000
    ),
    coverage_degraded INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(coverage_degraded) = 'integer' AND coverage_degraded IN (0, 1)
    ),
    unattributed_drop INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(unattributed_drop) = 'integer' AND unattributed_drop IN (0, 1)
    ),
    health_marker_count INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(health_marker_count) = 'integer'
        AND health_marker_count BETWEEN 0 AND 8
    ),
    health_set_digest TEXT NOT NULL CHECK (
        typeof(health_set_digest) = 'text'
        AND length(health_set_digest) = 64
        AND health_set_digest NOT GLOB '*[^0-9a-f]*'
    ),
    opened_at TEXT NOT NULL CHECK (
        typeof(opened_at) = 'text' AND length(opened_at) BETWEEN 20 AND 40
    ),
    updated_at TEXT NOT NULL CHECK (
        typeof(updated_at) = 'text' AND length(updated_at) BETWEEN 20 AND 40
    ),
    closed_at TEXT CHECK (
        closed_at IS NULL OR (
            typeof(closed_at) = 'text' AND length(closed_at) BETWEEN 20 AND 40
        )
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (connection_id, session_id),
    FOREIGN KEY (connection_id) REFERENCES connections(connection_id) ON DELETE CASCADE,
    CHECK ((state = 'closed') = (closed_at IS NOT NULL))
) WITHOUT ROWID;

CREATE INDEX capture_sessions_state_idx
    ON capture_sessions(connection_id, state, updated_at, session_id);

CREATE TABLE capture_events (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    receipt_ordinal INTEGER NOT NULL CHECK (
        typeof(receipt_ordinal) = 'integer' AND receipt_ordinal BETWEEN 1 AND 1000
    ),
    producer_event_digest TEXT NOT NULL CHECK (
        typeof(producer_event_digest) = 'text'
        AND length(producer_event_digest) = 64
        AND producer_event_digest NOT GLOB '*[^0-9a-f]*'
    ),
    event_kind TEXT NOT NULL CHECK (
        typeof(event_kind) = 'text'
        AND event_kind IN (
            'session_started',
            'action_started',
            'action_finished',
            'permission_denied',
            'subagent_started',
            'subagent_finished',
            'turn_finished',
            'controller_failed',
            'session_finished'
        )
    ),
    event_json BLOB NOT NULL CHECK (
        typeof(event_json) = 'blob' AND length(event_json) BETWEEN 1 AND 65536
    ),
    previous_event_tag TEXT CHECK (
        previous_event_tag IS NULL OR (
            typeof(previous_event_tag) = 'text'
            AND length(previous_event_tag) = 64
            AND previous_event_tag NOT GLOB '*[^0-9a-f]*'
        )
    ),
    event_tag TEXT NOT NULL CHECK (
        typeof(event_tag) = 'text'
        AND length(event_tag) = 64
        AND event_tag NOT GLOB '*[^0-9a-f]*'
    ),
    admission_source TEXT NOT NULL CHECK (
        typeof(admission_source) = 'text'
        AND admission_source IN ('direct', 'spool_drain')
    ),
    admitted_at TEXT NOT NULL CHECK (
        typeof(admitted_at) = 'text' AND length(admitted_at) BETWEEN 20 AND 40
    ),
    PRIMARY KEY (connection_id, session_id, receipt_ordinal),
    UNIQUE (connection_id, producer_event_digest),
    FOREIGN KEY (connection_id, session_id)
        REFERENCES capture_sessions(connection_id, session_id)
        ON DELETE CASCADE,
    CHECK (
        (receipt_ordinal = 1 AND previous_event_tag IS NULL)
        OR (receipt_ordinal > 1 AND previous_event_tag IS NOT NULL)
    )
) WITHOUT ROWID;

CREATE INDEX capture_events_session_idx
    ON capture_events(connection_id, session_id, receipt_ordinal);

CREATE TABLE capture_heads (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    receipt_count INTEGER NOT NULL CHECK (
        typeof(receipt_count) = 'integer' AND receipt_count BETWEEN 0 AND 1000
    ),
    head_event_tag TEXT CHECK (
        head_event_tag IS NULL OR (
            typeof(head_event_tag) = 'text'
            AND length(head_event_tag) = 64
            AND head_event_tag NOT GLOB '*[^0-9a-f]*'
        )
    ),
    head_tag TEXT NOT NULL CHECK (
        typeof(head_tag) = 'text'
        AND length(head_tag) = 64
        AND head_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (connection_id, session_id),
    FOREIGN KEY (connection_id, session_id)
        REFERENCES capture_sessions(connection_id, session_id)
        ON DELETE CASCADE,
    CHECK ((receipt_count = 0) = (head_event_tag IS NULL))
) WITHOUT ROWID;

CREATE TABLE capture_health (
    marker_id TEXT PRIMARY KEY CHECK (
        typeof(marker_id) = 'text'
        AND length(marker_id) = 64
        AND marker_id NOT GLOB '*[^0-9a-f]*'
    ),
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    code TEXT NOT NULL CHECK (
        typeof(code) = 'text'
        AND code IN (
            'producer_collision',
            'session_overflow',
            'spool_quota',
            'spool_unavailable',
            'unattributed_drop',
            'integrity_failure',
            'gap_detected',
            'coverage_degraded'
        )
    ),
    count INTEGER NOT NULL CHECK (typeof(count) = 'integer' AND count >= 1),
    lower_bound INTEGER NOT NULL CHECK (
        typeof(lower_bound) = 'integer' AND lower_bound IN (0, 1)
    ),
    created_at TEXT NOT NULL CHECK (
        typeof(created_at) = 'text' AND length(created_at) BETWEEN 20 AND 40
    ),
    updated_at TEXT NOT NULL CHECK (
        typeof(updated_at) = 'text' AND length(updated_at) BETWEEN 20 AND 40
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (connection_id) REFERENCES connections(connection_id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id, session_id)
        REFERENCES capture_sessions(connection_id, session_id)
        ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX capture_health_scope_idx
    ON capture_health(connection_id, session_id, code, updated_at);

CREATE TABLE feedback_labels (
    label_id TEXT PRIMARY KEY CHECK (
        typeof(label_id) = 'text'
        AND length(label_id) = 64
        AND label_id NOT GLOB '*[^0-9a-f]*'
    ),
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    label TEXT NOT NULL CHECK (
        typeof(label) = 'text'
        AND label IN ('memory-needed', 'not-memory-needed', 'uncertain')
    ),
    created_at TEXT NOT NULL CHECK (
        typeof(created_at) = 'text' AND length(created_at) BETWEEN 20 AND 40
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (connection_id, session_id)
        REFERENCES capture_sessions(connection_id, session_id)
        ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE deleted_sessions (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL CHECK (
        typeof(session_id) = 'text'
        AND length(session_id) = 64
        AND session_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_digest TEXT NOT NULL CHECK (
        typeof(project_digest) = 'text'
        AND length(project_digest) = 64
        AND project_digest NOT GLOB '*[^0-9a-f]*'
    ),
    deleted_at TEXT NOT NULL CHECK (
        typeof(deleted_at) = 'text' AND length(deleted_at) BETWEEN 20 AND 40
    ),
    tombstone_tag TEXT NOT NULL CHECK (
        typeof(tombstone_tag) = 'text'
        AND length(tombstone_tag) = 64
        AND tombstone_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (connection_id, session_id),
    FOREIGN KEY (connection_id) REFERENCES connections(connection_id) ON DELETE CASCADE
) WITHOUT ROWID;
