-- Performance indexes on large tables

-- crawl_data
CREATE INDEX idx_crawl_data_client_id ON crawl_data(client_id);
CREATE INDEX idx_crawl_data_url ON crawl_data(url);
CREATE INDEX idx_crawl_data_created_at ON crawl_data(created_at);

-- ga4_data
CREATE INDEX idx_ga4_data_client_id ON ga4_data(client_id);
CREATE INDEX idx_ga4_data_url ON ga4_data(url);
CREATE INDEX idx_ga4_data_created_at ON ga4_data(created_at);

-- gsc_data
CREATE INDEX idx_gsc_data_client_id ON gsc_data(client_id);
CREATE INDEX idx_gsc_data_url ON gsc_data(url);
CREATE INDEX idx_gsc_data_created_at ON gsc_data(created_at);

-- gbp_data
CREATE INDEX idx_gbp_data_client_id ON gbp_data(client_id);

-- backlink_data
CREATE INDEX idx_backlink_data_client_id ON backlink_data(client_id);
CREATE INDEX idx_backlink_data_target_url ON backlink_data(target_url);

-- wqa_results
CREATE INDEX idx_wqa_results_client_id ON wqa_results(client_id);
CREATE INDEX idx_wqa_results_url ON wqa_results(url);
CREATE INDEX idx_wqa_results_status ON wqa_results(status);
CREATE INDEX idx_wqa_results_created_at ON wqa_results(created_at);

-- keyword_clusters
CREATE INDEX idx_keyword_clusters_client_id ON keyword_clusters(client_id);
CREATE INDEX idx_keyword_clusters_status ON keyword_clusters(status);

-- content_queue
CREATE INDEX idx_content_queue_client_id ON content_queue(client_id);
CREATE INDEX idx_content_queue_status ON content_queue(status);
CREATE INDEX idx_content_queue_created_at ON content_queue(created_at);

-- schema_markup
CREATE INDEX idx_schema_markup_client_id ON schema_markup(client_id);
CREATE INDEX idx_schema_markup_url ON schema_markup(url);

-- asana_tasks
CREATE INDEX idx_asana_tasks_client_id ON asana_tasks(client_id);
CREATE INDEX idx_asana_tasks_status ON asana_tasks(status);

-- pipeline_runs
CREATE INDEX idx_pipeline_runs_client_id ON pipeline_runs(client_id);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_started_at ON pipeline_runs(started_at);

-- index_status
CREATE INDEX idx_index_status_client_id ON index_status(client_id);
CREATE INDEX idx_index_status_url ON index_status(url);
CREATE INDEX idx_index_status_checked_at ON index_status(checked_at);

-- monthly_reports
CREATE INDEX idx_monthly_reports_client_id ON monthly_reports(client_id);
