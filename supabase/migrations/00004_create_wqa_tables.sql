-- WQA results and operational tables

CREATE TABLE wqa_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    page_type TEXT,
    action TEXT,
    priority TEXT DEFAULT 'medium'
        CHECK (priority IN ('critical', 'high', 'medium', 'low')),
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped')),
    details JSONB DEFAULT '{}',
    excel_file_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE keyword_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    cluster_name TEXT NOT NULL,
    primary_keyword TEXT NOT NULL,
    keywords TEXT[] DEFAULT '{}',
    intent TEXT DEFAULT 'informational'
        CHECK (intent IN ('informational', 'navigational', 'transactional', 'commercial')),
    page_type TEXT,
    search_volume INTEGER,
    difficulty INTEGER,
    assigned_url TEXT,
    status TEXT DEFAULT 'identified'
        CHECK (status IN ('identified', 'assigned', 'content_created', 'published')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE content_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    keyword_cluster_id UUID REFERENCES keyword_clusters(id),
    title TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content_body TEXT,
    meta_title TEXT,
    meta_description TEXT,
    schema_markup JSONB,
    quality_score INTEGER,
    status TEXT DEFAULT 'draft'
        CHECK (status IN ('draft', 'review', 'approved', 'published', 'rejected')),
    cms_post_id TEXT,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE content_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    page_type TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    structure JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE schema_markup (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    schema_type TEXT NOT NULL,
    json_ld JSONB NOT NULL DEFAULT '{}',
    status TEXT DEFAULT 'generated'
        CHECK (status IN ('generated', 'applied', 'verified')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE asana_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    asana_task_id TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    section TEXT NOT NULL,
    status TEXT DEFAULT 'open'
        CHECK (status IN ('open', 'completed')),
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE monthly_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    report_month DATE NOT NULL,
    metrics_snapshot JSONB DEFAULT '{}',
    executive_summary TEXT,
    anomalies JSONB DEFAULT '[]',
    pdf_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    pipeline_type TEXT NOT NULL DEFAULT 'full',
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'success', 'partial', 'cancelled')),
    workers_requested TEXT[] DEFAULT '{}',
    worker_results JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    error TEXT,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE index_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    is_indexed BOOLEAN DEFAULT false,
    last_crawled TIMESTAMPTZ,
    coverage_state TEXT,
    verdict TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Auto-update triggers
CREATE TRIGGER wqa_results_updated_at
    BEFORE UPDATE ON wqa_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER content_queue_updated_at
    BEFORE UPDATE ON content_queue
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
