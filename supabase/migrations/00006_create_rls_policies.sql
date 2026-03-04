-- Row Level Security policies
-- Users see only their own clients; service_role has full access

-- Enable RLS on all tables
ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_api_credentials ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE crawl_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE crawl_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE ga4_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE gsc_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE gbp_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE backlink_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE wqa_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE schema_markup ENABLE ROW LEVEL SECURITY;
ALTER TABLE asana_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE monthly_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE index_status ENABLE ROW LEVEL SECURITY;

-- Clients: owner can CRUD their own
CREATE POLICY clients_owner_select ON clients
    FOR SELECT USING (owner_user_id = auth.uid());
CREATE POLICY clients_owner_insert ON clients
    FOR INSERT WITH CHECK (owner_user_id = auth.uid());
CREATE POLICY clients_owner_update ON clients
    FOR UPDATE USING (owner_user_id = auth.uid());
CREATE POLICY clients_owner_delete ON clients
    FOR DELETE USING (owner_user_id = auth.uid());

-- Helper function: check if user owns the client
CREATE OR REPLACE FUNCTION user_owns_client(cid UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM clients WHERE id = cid AND owner_user_id = auth.uid()
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Macro for client-scoped tables: user sees rows where they own the client
-- client_api_credentials
CREATE POLICY cac_select ON client_api_credentials FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY cac_insert ON client_api_credentials FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY cac_update ON client_api_credentials FOR UPDATE USING (user_owns_client(client_id));
CREATE POLICY cac_delete ON client_api_credentials FOR DELETE USING (user_owns_client(client_id));

-- business_rules
CREATE POLICY br_select ON business_rules FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY br_insert ON business_rules FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY br_update ON business_rules FOR UPDATE USING (user_owns_client(client_id));
CREATE POLICY br_delete ON business_rules FOR DELETE USING (user_owns_client(client_id));

-- crawl_runs
CREATE POLICY cr_select ON crawl_runs FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY cr_insert ON crawl_runs FOR INSERT WITH CHECK (user_owns_client(client_id));

-- crawl_data
CREATE POLICY cd_select ON crawl_data FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY cd_insert ON crawl_data FOR INSERT WITH CHECK (user_owns_client(client_id));

-- ga4_data
CREATE POLICY ga4_select ON ga4_data FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY ga4_insert ON ga4_data FOR INSERT WITH CHECK (user_owns_client(client_id));

-- gsc_data
CREATE POLICY gsc_select ON gsc_data FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY gsc_insert ON gsc_data FOR INSERT WITH CHECK (user_owns_client(client_id));

-- gbp_data
CREATE POLICY gbp_select ON gbp_data FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY gbp_insert ON gbp_data FOR INSERT WITH CHECK (user_owns_client(client_id));

-- backlink_data
CREATE POLICY bl_select ON backlink_data FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY bl_insert ON backlink_data FOR INSERT WITH CHECK (user_owns_client(client_id));

-- wqa_results
CREATE POLICY wqa_select ON wqa_results FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY wqa_insert ON wqa_results FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY wqa_update ON wqa_results FOR UPDATE USING (user_owns_client(client_id));

-- keyword_clusters
CREATE POLICY kc_select ON keyword_clusters FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY kc_insert ON keyword_clusters FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY kc_update ON keyword_clusters FOR UPDATE USING (user_owns_client(client_id));

-- content_queue
CREATE POLICY cq_select ON content_queue FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY cq_insert ON content_queue FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY cq_update ON content_queue FOR UPDATE USING (user_owns_client(client_id));

-- content_templates (readable by all authenticated, writable by service_role only)
CREATE POLICY ct_select ON content_templates FOR SELECT USING (auth.role() = 'authenticated');

-- schema_markup
CREATE POLICY sm_select ON schema_markup FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY sm_insert ON schema_markup FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY sm_update ON schema_markup FOR UPDATE USING (user_owns_client(client_id));

-- asana_tasks
CREATE POLICY at_select ON asana_tasks FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY at_insert ON asana_tasks FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY at_update ON asana_tasks FOR UPDATE USING (user_owns_client(client_id));

-- monthly_reports
CREATE POLICY mr_select ON monthly_reports FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY mr_insert ON monthly_reports FOR INSERT WITH CHECK (user_owns_client(client_id));

-- pipeline_runs
CREATE POLICY pr_select ON pipeline_runs FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY pr_insert ON pipeline_runs FOR INSERT WITH CHECK (user_owns_client(client_id));
CREATE POLICY pr_update ON pipeline_runs FOR UPDATE USING (user_owns_client(client_id));

-- index_status
CREATE POLICY is_select ON index_status FOR SELECT USING (user_owns_client(client_id));
CREATE POLICY is_insert ON index_status FOR INSERT WITH CHECK (user_owns_client(client_id));
