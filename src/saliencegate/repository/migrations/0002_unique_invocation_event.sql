DELETE FROM projection_decisions
WHERE rowid NOT IN (
    SELECT MIN(rowid)
    FROM projection_decisions
    GROUP BY run_id, event_sequence
);

CREATE UNIQUE INDEX projection_decisions_authoritative_event_idx
    ON projection_decisions(run_id, event_sequence);
