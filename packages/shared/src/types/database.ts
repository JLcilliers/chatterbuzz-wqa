// Auto-generated types placeholder — run `pnpm generate-types` after Supabase setup
// Manual types below match the Phase 1 migration schema

export interface Client {
  id: string;
  name: string;
  domain: string;
  industry: string;
  business_type: string;
  target_locations: string[];
  goals: Record<string, unknown>;
  brand_voice: Record<string, unknown>;
  status: 'onboarding' | 'active' | 'paused' | 'churned';
  created_at: string;
  updated_at: string;
  owner_user_id: string;
}

export interface ClientApiCredential {
  id: string;
  client_id: string;
  provider: 'google' | 'wordpress' | 'webflow' | 'asana' | 'gbp';
  credentials_encrypted: string;
  scopes: string[];
  expires_at: string | null;
  created_at: string;
}

export interface BusinessRule {
  id: string;
  client_id: string;
  rule_type: string;
  rule_config: Record<string, unknown>;
  created_at: string;
}

export interface CrawlRun {
  id: string;
  client_id: string;
  source: 'screaming_frog' | 'sitebulb' | 'custom';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  row_count: number;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface CrawlData {
  id: string;
  crawl_run_id: string;
  client_id: string;
  url: string;
  status_code: number | null;
  title: string | null;
  meta_description: string | null;
  h1: string | null;
  word_count: number | null;
  indexability: string | null;
  canonical: string | null;
  page_type: string | null;
  crawl_depth: number | null;
  internal_links_in: number | null;
  internal_links_out: number | null;
  load_time: number | null;
  created_at: string;
}

export interface GA4Data {
  id: string;
  client_id: string;
  url: string;
  sessions: number;
  engaged_sessions: number;
  engagement_rate: number | null;
  bounce_rate: number | null;
  avg_session_duration: number | null;
  conversions: number;
  revenue: number | null;
  date_range_start: string;
  date_range_end: string;
  created_at: string;
}

export interface GSCData {
  id: string;
  client_id: string;
  url: string;
  query: string | null;
  clicks: number;
  impressions: number;
  ctr: number | null;
  position: number | null;
  date_range_start: string;
  date_range_end: string;
  created_at: string;
}

export interface GBPData {
  id: string;
  client_id: string;
  location_id: string;
  metric_type: string;
  value: number;
  period_start: string;
  period_end: string;
  created_at: string;
}

export interface BacklinkData {
  id: string;
  client_id: string;
  target_url: string;
  source_url: string;
  anchor_text: string | null;
  domain_rating: number | null;
  is_dofollow: boolean;
  first_seen: string | null;
  created_at: string;
}

export interface WQAResult {
  id: string;
  client_id: string;
  url: string;
  page_type: string | null;
  action: string | null;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  details: Record<string, unknown>;
  excel_file_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface KeywordCluster {
  id: string;
  client_id: string;
  cluster_name: string;
  primary_keyword: string;
  keywords: string[];
  intent: 'informational' | 'navigational' | 'transactional' | 'commercial';
  page_type: string | null;
  search_volume: number | null;
  difficulty: number | null;
  assigned_url: string | null;
  status: 'identified' | 'assigned' | 'content_created' | 'published';
  created_at: string;
}

export interface ContentQueueItem {
  id: string;
  client_id: string;
  keyword_cluster_id: string | null;
  title: string;
  content_type: string;
  content_body: string | null;
  meta_title: string | null;
  meta_description: string | null;
  schema_markup: string | null;
  quality_score: number | null;
  status: 'draft' | 'review' | 'approved' | 'published' | 'rejected';
  cms_post_id: string | null;
  published_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ContentTemplate {
  id: string;
  name: string;
  page_type: string;
  system_prompt: string;
  structure: Record<string, unknown>;
  created_at: string;
}

export interface SchemaMarkup {
  id: string;
  client_id: string;
  url: string;
  schema_type: string;
  json_ld: Record<string, unknown>;
  status: 'generated' | 'applied' | 'verified';
  created_at: string;
}

export interface AsanaTask {
  id: string;
  client_id: string;
  asana_task_id: string;
  source_type: string;
  source_id: string;
  section: string;
  status: 'open' | 'completed';
  synced_at: string;
  created_at: string;
}

export interface MonthlyReport {
  id: string;
  client_id: string;
  report_month: string;
  metrics_snapshot: Record<string, unknown>;
  executive_summary: string | null;
  anomalies: Record<string, unknown>[];
  pdf_url: string | null;
  created_at: string;
}

export interface PipelineRun {
  id: string;
  client_id: string;
  pipeline_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
  metadata: Record<string, unknown>;
}

export interface IndexStatus {
  id: string;
  client_id: string;
  url: string;
  is_indexed: boolean;
  last_crawled: string | null;
  coverage_state: string | null;
  verdict: string | null;
  checked_at: string;
  created_at: string;
}

// Helper: make all fields optional for Insert/Update
type TableDef<T> = { Row: T; Insert: Partial<T>; Update: Partial<T>; Relationships: [] };

// Database schema aggregate type
export interface Database {
  public: {
    Tables: {
      clients: TableDef<Client>;
      client_api_credentials: TableDef<ClientApiCredential>;
      business_rules: TableDef<BusinessRule>;
      crawl_runs: TableDef<CrawlRun>;
      crawl_data: TableDef<CrawlData>;
      ga4_data: TableDef<GA4Data>;
      gsc_data: TableDef<GSCData>;
      gbp_data: TableDef<GBPData>;
      backlink_data: TableDef<BacklinkData>;
      wqa_results: TableDef<WQAResult>;
      keyword_clusters: TableDef<KeywordCluster>;
      content_queue: TableDef<ContentQueueItem>;
      content_templates: TableDef<ContentTemplate>;
      schema_markup: TableDef<SchemaMarkup>;
      asana_tasks: TableDef<AsanaTask>;
      monthly_reports: TableDef<MonthlyReport>;
      pipeline_runs: TableDef<PipelineRun>;
      index_status: TableDef<IndexStatus>;
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
    CompositeTypes: Record<string, never>;
  };
}
