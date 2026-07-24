CREATE TABLE capture_global_parents (
    global_parent_id TEXT PRIMARY KEY CHECK (
        typeof(global_parent_id) = 'text'
        AND length(global_parent_id) = 52
        AND substr(global_parent_id, 1, 4) = 'sgg-'
        AND substr(global_parent_id, 5) NOT GLOB '*[^0-9a-f]*'
    ),
    provider_id TEXT NOT NULL CHECK (
        typeof(provider_id) = 'text'
        AND provider_id IN ('codex', 'claude-code', 'opencode', 'pi')
    ),
    config_root_digest TEXT NOT NULL CHECK (
        typeof(config_root_digest) = 'text'
        AND length(config_root_digest) = 64
        AND config_root_digest NOT GLOB '*[^0-9a-f]*'
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
    generation INTEGER NOT NULL CHECK (
        typeof(generation) = 'integer' AND generation BETWEEN 1 AND 1000000
    ),
    state TEXT NOT NULL CHECK (
        typeof(state) = 'text'
        AND state IN ('pending', 'enabled', 'draining', 'disabled', 'deleting')
    ),
    health_marker_count INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(health_marker_count) = 'integer'
        AND health_marker_count BETWEEN 0 AND 5
    ),
    health_set_digest TEXT NOT NULL CHECK (
        typeof(health_set_digest) = 'text'
        AND length(health_set_digest) = 64
        AND health_set_digest NOT GLOB '*[^0-9a-f]*'
    ),
    exclusion_count INTEGER NOT NULL DEFAULT 0 CHECK (
        typeof(exclusion_count) = 'integer'
        AND exclusion_count BETWEEN 0 AND 1000
    ),
    exclusion_set_digest TEXT NOT NULL CHECK (
        typeof(exclusion_set_digest) = 'text'
        AND length(exclusion_set_digest) = 64
        AND exclusion_set_digest NOT GLOB '*[^0-9a-f]*'
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
    UNIQUE (provider_id, config_root_digest, generation)
) WITHOUT ROWID;

CREATE INDEX capture_global_parents_provider_idx
    ON capture_global_parents(provider_id, state, global_parent_id);

CREATE TABLE capture_global_children (
    global_parent_id TEXT NOT NULL,
    connection_id TEXT NOT NULL UNIQUE,
    project_digest TEXT NOT NULL CHECK (
        typeof(project_digest) = 'text'
        AND length(project_digest) = 64
        AND project_digest NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL CHECK (
        typeof(created_at) = 'text' AND length(created_at) BETWEEN 20 AND 40
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (global_parent_id, project_digest),
    FOREIGN KEY (global_parent_id)
        REFERENCES capture_global_parents(global_parent_id),
    FOREIGN KEY (connection_id)
        REFERENCES connections(connection_id)
        ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX capture_global_children_connection_idx
    ON capture_global_children(connection_id, global_parent_id);

CREATE TABLE capture_global_exclusions (
    global_parent_id TEXT NOT NULL,
    project_digest TEXT NOT NULL CHECK (
        typeof(project_digest) = 'text'
        AND length(project_digest) = 64
        AND project_digest NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL CHECK (
        typeof(created_at) = 'text' AND length(created_at) BETWEEN 20 AND 40
    ),
    row_tag TEXT NOT NULL CHECK (
        typeof(row_tag) = 'text'
        AND length(row_tag) = 64
        AND row_tag NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (global_parent_id, project_digest),
    FOREIGN KEY (global_parent_id)
        REFERENCES capture_global_parents(global_parent_id)
        ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX capture_global_exclusions_scope_idx
    ON capture_global_exclusions(global_parent_id, project_digest);

CREATE TABLE capture_global_health (
    marker_id TEXT PRIMARY KEY CHECK (
        typeof(marker_id) = 'text'
        AND length(marker_id) = 64
        AND marker_id NOT GLOB '*[^0-9a-f]*'
    ),
    global_parent_id TEXT NOT NULL,
    code TEXT NOT NULL CHECK (
        typeof(code) = 'text'
        AND code IN (
            'unknown_child_event',
            'project_identity_unavailable',
            'enrollment_rejected',
            'child_limit_reached',
            'project_excluded'
        )
    ),
    count INTEGER NOT NULL CHECK (
        typeof(count) = 'integer' AND count BETWEEN 1 AND 1000000
    ),
    saturated INTEGER NOT NULL CHECK (
        typeof(saturated) = 'integer' AND saturated IN (0, 1)
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
    UNIQUE (global_parent_id, code),
    FOREIGN KEY (global_parent_id)
        REFERENCES capture_global_parents(global_parent_id)
        ON DELETE CASCADE,
    CHECK ((count = 1000000) = (saturated = 1))
) WITHOUT ROWID;

CREATE INDEX capture_global_health_scope_idx
    ON capture_global_health(global_parent_id, code, updated_at);
