ALTER TABLE capture_sessions
    ADD COLUMN transport_required INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(transport_required) = 'integer' AND transport_required IN (0, 1)
    );

ALTER TABLE capture_sessions
    ADD COLUMN transport_head_tag TEXT CHECK (
        transport_head_tag IS NULL OR (
            typeof(transport_head_tag) = 'text'
            AND length(transport_head_tag) = 64
            AND transport_head_tag NOT GLOB '*[^0-9a-f]*'
        )
    );

CREATE TABLE capture_transport_heads (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    receipt_count INTEGER NOT NULL CHECK (
        typeof(receipt_count) = 'integer' AND receipt_count BETWEEN 0 AND 1000
    ),
    head_receipt_tag TEXT CHECK (
        head_receipt_tag IS NULL OR (
            typeof(head_receipt_tag) = 'text'
            AND length(head_receipt_tag) = 64
            AND head_receipt_tag NOT GLOB '*[^0-9a-f]*'
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
    CHECK ((receipt_count = 0) = (head_receipt_tag IS NULL))
) WITHOUT ROWID;

CREATE TABLE capture_transport_receipts (
    connection_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    transport_ordinal INTEGER NOT NULL CHECK (
        typeof(transport_ordinal) = 'integer'
        AND transport_ordinal BETWEEN 1 AND 1000
    ),
    batch_ref TEXT NOT NULL CHECK (
        typeof(batch_ref) = 'text'
        AND length(batch_ref) = 64
        AND batch_ref NOT GLOB '*[^0-9a-f]*'
    ),
    chunk_index INTEGER NOT NULL CHECK (
        typeof(chunk_index) = 'integer'
        AND chunk_index BETWEEN 0 AND 999
    ),
    chunk_count INTEGER NOT NULL CHECK (
        typeof(chunk_count) = 'integer'
        AND chunk_count BETWEEN 1 AND 1000
    ),
    chunk_digest TEXT NOT NULL CHECK (
        typeof(chunk_digest) = 'text'
        AND length(chunk_digest) = 64
        AND chunk_digest NOT GLOB '*[^0-9a-f]*'
    ),
    intake_count INTEGER NOT NULL CHECK (
        typeof(intake_count) = 'integer'
        AND intake_count BETWEEN 0 AND 1000
    ),
    intake_set_digest TEXT NOT NULL CHECK (
        typeof(intake_set_digest) = 'text'
        AND length(intake_set_digest) = 64
        AND intake_set_digest NOT GLOB '*[^0-9a-f]*'
    ),
    post_event_count INTEGER NOT NULL CHECK (
        typeof(post_event_count) = 'integer'
        AND post_event_count BETWEEN 0 AND 1000
    ),
    post_head_event_tag TEXT CHECK (
        post_head_event_tag IS NULL OR (
            typeof(post_head_event_tag) = 'text'
            AND length(post_head_event_tag) = 64
            AND post_head_event_tag NOT GLOB '*[^0-9a-f]*'
        )
    ),
    previous_receipt_tag TEXT CHECK (
        previous_receipt_tag IS NULL OR (
            typeof(previous_receipt_tag) = 'text'
            AND length(previous_receipt_tag) = 64
            AND previous_receipt_tag NOT GLOB '*[^0-9a-f]*'
        )
    ),
    receipt_tag TEXT NOT NULL CHECK (
        typeof(receipt_tag) = 'text'
        AND length(receipt_tag) = 64
        AND receipt_tag NOT GLOB '*[^0-9a-f]*'
    ),
    admitted_at TEXT NOT NULL CHECK (
        typeof(admitted_at) = 'text' AND length(admitted_at) BETWEEN 20 AND 40
    ),
    PRIMARY KEY (connection_id, session_id, transport_ordinal),
    UNIQUE (connection_id, batch_ref, chunk_index),
    FOREIGN KEY (connection_id, session_id)
        REFERENCES capture_sessions(connection_id, session_id)
        ON DELETE CASCADE,
    CHECK (chunk_index < chunk_count),
    CHECK ((post_event_count = 0) = (post_head_event_tag IS NULL)),
    CHECK (
        (transport_ordinal = 1 AND previous_receipt_tag IS NULL)
        OR (transport_ordinal > 1 AND previous_receipt_tag IS NOT NULL)
    )
) WITHOUT ROWID;

CREATE INDEX capture_transport_batch_idx
    ON capture_transport_receipts(connection_id, session_id, batch_ref, chunk_index);
