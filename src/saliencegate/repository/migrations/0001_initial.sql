PRAGMA application_id = 0x534C4754;

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version >= 1),
    name TEXT NOT NULL UNIQUE CHECK (length(name) BETWEEN 1 AND 100),
    checksum TEXT NOT NULL CHECK (
        length(checksum) = 64
        AND checksum NOT GLOB '*[^0-9a-f]*'
    ),
    applied_at TEXT NOT NULL CHECK (length(applied_at) >= 20)
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY CHECK (length(run_id) = 36),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20)
) WITHOUT ROWID;

CREATE TABLE ledger_entries (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    position INTEGER NOT NULL CHECK (position >= 1),
    record_key TEXT NOT NULL CHECK (length(record_key) BETWEEN 1 AND 200),
    record_type TEXT NOT NULL CHECK (
        record_type IN (
            'trace_event',
            'signal',
            'invocation_decision',
            'cycle_record',
            'intervention_outcome',
            'delivery_record'
        )
    ),
    entry_json BLOB NOT NULL CHECK (
        typeof(entry_json) = 'blob'
        AND length(entry_json) > 0
    ),
    record_algorithm TEXT NOT NULL CHECK (
        record_algorithm IN ('hmac_sha256', 'synthetic_sha256')
    ),
    record_tag TEXT NOT NULL CHECK (
        length(record_tag) = 64
        AND record_tag NOT GLOB '*[^0-9a-f]*'
    ),
    previous_chain_algorithm TEXT CHECK (
        previous_chain_algorithm IS NULL
        OR previous_chain_algorithm IN ('hmac_sha256', 'synthetic_sha256')
    ),
    previous_chain_tag TEXT CHECK (
        previous_chain_tag IS NULL
        OR (
            length(previous_chain_tag) = 64
            AND previous_chain_tag NOT GLOB '*[^0-9a-f]*'
        )
    ),
    chain_algorithm TEXT NOT NULL CHECK (
        chain_algorithm IN ('hmac_sha256', 'synthetic_sha256')
    ),
    chain_tag TEXT NOT NULL CHECK (
        length(chain_tag) = 64
        AND chain_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (run_id, position),
    UNIQUE (run_id, record_key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    CHECK (
        (position = 1 AND previous_chain_algorithm IS NULL AND previous_chain_tag IS NULL)
        OR (
            position > 1
            AND previous_chain_algorithm IS NOT NULL
            AND previous_chain_tag IS NOT NULL
        )
    ),
    CHECK (record_algorithm = chain_algorithm),
    CHECK (
        previous_chain_algorithm IS NULL
        OR previous_chain_algorithm = chain_algorithm
    )
) WITHOUT ROWID;

CREATE INDEX ledger_entries_record_type_idx
    ON ledger_entries(run_id, record_type, position);

CREATE TABLE ledger_heads (
    run_id TEXT PRIMARY KEY CHECK (length(run_id) = 36),
    entry_count INTEGER NOT NULL CHECK (entry_count >= 1),
    algorithm TEXT NOT NULL CHECK (
        algorithm IN ('hmac_sha256', 'synthetic_sha256')
    ),
    chain_tag TEXT NOT NULL CHECK (
        length(chain_tag) = 64
        AND chain_tag NOT GLOB '*[^0-9a-f]*'
    ),
    projection_tag TEXT NOT NULL CHECK (
        length(projection_tag) = 64
        AND projection_tag NOT GLOB '*[^0-9a-f]*'
    ),
    head_tag TEXT NOT NULL CHECK (
        length(head_tag) = 64
        AND head_tag NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, entry_count)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
) WITHOUT ROWID;

CREATE TABLE projection_state (
    run_id TEXT PRIMARY KEY CHECK (length(run_id) = 36),
    state_schema_version INTEGER NOT NULL DEFAULT 1 CHECK (state_schema_version >= 1),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    ingestion_cursor INTEGER NOT NULL CHECK (ingestion_cursor >= 0),
    memory_cursor INTEGER NOT NULL CHECK (
        memory_cursor >= 0
        AND memory_cursor <= ingestion_cursor
    ),
    current_private_status_id TEXT CHECK (
        current_private_status_id IS NULL
        OR length(current_private_status_id) = 36
    ),
    projection_digests_json BLOB NOT NULL CHECK (
        typeof(projection_digests_json) = 'blob'
        AND length(projection_digests_json) > 0
    ),
    state_algorithm TEXT NOT NULL CHECK (
        state_algorithm IN ('hmac_sha256', 'synthetic_sha256')
    ),
    state_tag TEXT NOT NULL CHECK (
        length(state_tag) = 64
        AND state_tag NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
) WITHOUT ROWID;

CREATE TABLE projection_events (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    event_id TEXT NOT NULL CHECK (length(event_id) = 36),
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    source_event_id TEXT NOT NULL CHECK (length(source_event_id) BETWEEN 1 AND 256),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, event_id),
    UNIQUE (run_id, sequence),
    UNIQUE (run_id, source_event_id),
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE projection_signals (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    signal_id TEXT NOT NULL CHECK (length(signal_id) = 36),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, signal_id),
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX projection_signals_created_idx
    ON projection_signals(run_id, created_at, signal_id);

CREATE TABLE projection_decisions (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    decision_id TEXT NOT NULL CHECK (length(decision_id) = 36),
    event_sequence INTEGER NOT NULL CHECK (event_sequence >= 1),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, decision_id),
    FOREIGN KEY (run_id, event_sequence)
        REFERENCES projection_events(run_id, sequence)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX projection_decisions_event_idx
    ON projection_decisions(run_id, event_sequence, created_at);

CREATE TABLE projection_cycle_revisions (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    cycle_id TEXT NOT NULL CHECK (
        length(cycle_id) = 64
        AND cycle_id NOT GLOB '*[^0-9a-f]*'
    ),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'reserved', 'running', 'committed', 'failed')
    ),
    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    invocation_decision_id TEXT NOT NULL CHECK (length(invocation_decision_id) = 36),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    updated_at TEXT NOT NULL CHECK (length(updated_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, cycle_id, revision),
    FOREIGN KEY (run_id, invocation_decision_id)
        REFERENCES projection_decisions(run_id, decision_id)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX projection_cycle_latest_idx
    ON projection_cycle_revisions(run_id, cycle_id)
    WHERE is_latest = 1;

CREATE INDEX projection_cycle_state_idx
    ON projection_cycle_revisions(run_id, state, is_latest, updated_at);

CREATE TABLE projection_memories (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    memory_id TEXT NOT NULL CHECK (length(memory_id) = 36),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    kind TEXT NOT NULL CHECK (kind IN ('private_status', 'knowledge', 'procedural')),
    validity TEXT NOT NULL CHECK (
        validity IN ('active', 'invalidated', 'expired', 'superseded')
    ),
    trust_label TEXT NOT NULL CHECK (
        trust_label IN (
            'trusted_runtime',
            'trusted_controller',
            'untrusted_task_input',
            'untrusted_tool_output',
            'untrusted_model_output',
            'untrusted_external_memory',
            'synthetic_fixture'
        )
    ),
    content TEXT NOT NULL CHECK (length(content) >= 1),
    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
    source_cycle_id TEXT NOT NULL CHECK (
        length(source_cycle_id) = 64
        AND source_cycle_id NOT GLOB '*[^0-9a-f]*'
    ),
    source_cycle_revision INTEGER NOT NULL CHECK (source_cycle_revision >= 1),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, memory_id, revision),
    FOREIGN KEY (run_id, source_cycle_id, source_cycle_revision)
        REFERENCES projection_cycle_revisions(run_id, cycle_id, revision)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX projection_memory_latest_idx
    ON projection_memories(run_id, memory_id)
    WHERE is_latest = 1;

CREATE INDEX projection_memory_filter_idx
    ON projection_memories(run_id, is_latest, validity, kind, trust_label, memory_id);

CREATE VIEW projection_active_memories AS
SELECT rowid, content, run_id, memory_id, revision
FROM projection_memories
WHERE is_latest = 1 AND validity = 'active';

CREATE TABLE projection_interventions (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    intervention_id TEXT NOT NULL CHECK (length(intervention_id) = 36),
    cycle_id TEXT NOT NULL CHECK (
        length(cycle_id) = 64
        AND cycle_id NOT GLOB '*[^0-9a-f]*'
    ),
    cycle_revision INTEGER NOT NULL CHECK (cycle_revision >= 1),
    action TEXT NOT NULL CHECK (action IN ('silence', 'remind')),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, intervention_id),
    UNIQUE (run_id, cycle_id),
    FOREIGN KEY (run_id, cycle_id, cycle_revision)
        REFERENCES projection_cycle_revisions(run_id, cycle_id, revision)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX projection_interventions_created_idx
    ON projection_interventions(run_id, created_at, intervention_id);

CREATE TABLE projection_outcomes (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    outcome_id TEXT NOT NULL CHECK (length(outcome_id) = 36),
    intervention_id TEXT NOT NULL CHECK (length(intervention_id) = 36),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, outcome_id),
    FOREIGN KEY (run_id, intervention_id)
        REFERENCES projection_interventions(run_id, intervention_id)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX projection_outcomes_intervention_idx
    ON projection_outcomes(run_id, intervention_id, created_at);

CREATE TABLE projection_delivery_revisions (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    delivery_id TEXT NOT NULL CHECK (length(delivery_id) = 36),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    intervention_id TEXT NOT NULL CHECK (length(intervention_id) = 36),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'claimed', 'attempting', 'delivered', 'failed', 'unknown', 'rejected')
    ),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
    ledger_position INTEGER NOT NULL CHECK (ledger_position >= 1),
    created_at TEXT NOT NULL CHECK (length(created_at) >= 20),
    updated_at TEXT NOT NULL CHECK (length(updated_at) >= 20),
    record_json BLOB NOT NULL CHECK (
        typeof(record_json) = 'blob'
        AND length(record_json) > 0
    ),
    PRIMARY KEY (run_id, delivery_id, revision),
    FOREIGN KEY (run_id, intervention_id)
        REFERENCES projection_interventions(run_id, intervention_id)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (run_id, ledger_position)
        REFERENCES ledger_entries(run_id, position)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE UNIQUE INDEX projection_delivery_latest_idx
    ON projection_delivery_revisions(run_id, delivery_id)
    WHERE is_latest = 1;

CREATE INDEX projection_delivery_outbox_idx
    ON projection_delivery_revisions(run_id, state, is_latest, updated_at, delivery_id);

CREATE TABLE projection_budgets (
    run_id TEXT NOT NULL CHECK (length(run_id) = 36),
    cycle_id TEXT NOT NULL CHECK (
        length(cycle_id) = 64
        AND cycle_id NOT GLOB '*[^0-9a-f]*'
    ),
    cycle_revision INTEGER NOT NULL CHECK (cycle_revision >= 1),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'reserved', 'running', 'committed', 'failed')
    ),
    reservation_json BLOB CHECK (
        reservation_json IS NULL
        OR (typeof(reservation_json) = 'blob' AND length(reservation_json) > 0)
    ),
    settlement_json BLOB CHECK (
        settlement_json IS NULL
        OR (typeof(settlement_json) = 'blob' AND length(settlement_json) > 0)
    ),
    PRIMARY KEY (run_id, cycle_id),
    FOREIGN KEY (run_id, cycle_id, cycle_revision)
        REFERENCES projection_cycle_revisions(run_id, cycle_id, revision)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
) WITHOUT ROWID;

CREATE INDEX projection_budget_state_idx
    ON projection_budgets(run_id, state, cycle_id);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    content,
    run_id UNINDEXED,
    memory_id UNINDEXED,
    revision UNINDEXED,
    content = 'projection_active_memories',
    content_rowid = 'rowid',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER projection_memories_fts_insert
AFTER INSERT ON projection_memories
WHEN NEW.is_latest = 1 AND NEW.validity = 'active'
BEGIN
    INSERT INTO memory_fts(rowid, content, run_id, memory_id, revision)
    VALUES (NEW.rowid, NEW.content, NEW.run_id, NEW.memory_id, NEW.revision);
END;

CREATE TRIGGER projection_memories_fts_delete
AFTER DELETE ON projection_memories
WHEN OLD.is_latest = 1 AND OLD.validity = 'active'
BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, run_id, memory_id, revision)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.run_id, OLD.memory_id, OLD.revision);
END;

CREATE TRIGGER projection_memories_fts_update
AFTER UPDATE ON projection_memories
BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, run_id, memory_id, revision)
    SELECT 'delete', OLD.rowid, OLD.content, OLD.run_id, OLD.memory_id, OLD.revision
    WHERE OLD.is_latest = 1 AND OLD.validity = 'active';

    INSERT INTO memory_fts(rowid, content, run_id, memory_id, revision)
    SELECT NEW.rowid, NEW.content, NEW.run_id, NEW.memory_id, NEW.revision
    WHERE NEW.is_latest = 1 AND NEW.validity = 'active';
END;
