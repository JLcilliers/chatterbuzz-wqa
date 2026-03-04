-- Data ingestion tables

CREATE TABLE crawl_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'screaming_frog'
        CHECK (source IN ('screaming_frog', 'sitebulb', 'custom')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    row_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE crawl_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    crawl_run_id UUID NOT NULL REFERENCES crawl_runs(id) ON DELETE CASCADE,
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status_code INTEGER,
    title TEXT,
    meta_description TEXT,
    h1 TEXT,
    word_count INTEGER,
    indexability TEXT,
    canonical TEXT,
    page_type TEXT,
    crawl_depth INTEGER,
    internal_links_in INTEGER,
    internal_links_out INTEGER,
    load_time NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ga4_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    sessions INTEGER DEFAULT 0,
    engaged_sessions INTEGER DEFAULT 0,
    engagement_rate NUMERIC,
    bounce_rate NUMERIC,
    avg_session_duration NUMERIC,
    conversions INTEGER DEFAULT 0,
    revenue NUMERIC,
    date_range_start DATE NOT NULL,
    date_range_end DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE gsc_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    query TEXT,
    clicks INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    ctr NUMERIC,
    position NUMERIC,
    date_range_start DATE NOT NULL,
    date_range_end DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE gbp_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    location_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value NUMERIC DEFAULT 0,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE backlink_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    target_url TEXT NOT NULL,
    source_url TEXT NOT NULL,
    anchor_text TEXT,
    domain_rating NUMERIC,
    is_dofollow BOOLEAN DEFAULT true,
    first_seen DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
