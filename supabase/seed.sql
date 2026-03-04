-- Seed data: demo client for development/testing
-- Note: owner_user_id should be set to a real auth.users ID after signup

INSERT INTO clients (id, name, domain, industry, business_type, target_locations, goals, brand_voice, status)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Demo Client — Davidson Homes',
    'davidsonhomes.com',
    'Real Estate',
    'Home Builder',
    ARRAY['Charlotte, NC', 'Raleigh, NC', 'Nashville, TN'],
    '{"primary": "increase organic traffic", "secondary": "improve local visibility", "kpi_targets": {"organic_sessions_growth": 20, "keyword_rankings_top10": 50}}'::jsonb,
    '{"tone": "professional", "style": "authoritative", "avoid": ["jargon", "aggressive sales language"]}'::jsonb,
    'active'
);

-- Demo business rules
INSERT INTO business_rules (client_id, rule_type, rule_config) VALUES
('00000000-0000-0000-0000-000000000001', 'content_length_minimum', '{"min_words": 800}'::jsonb),
('00000000-0000-0000-0000-000000000001', 'publish_rate_limit', '{"max_per_week": 100}'::jsonb),
('00000000-0000-0000-0000-000000000001', 'priority_pages', '{"patterns": ["/communities/*", "/floor-plans/*"]}'::jsonb);

-- Demo content template
INSERT INTO content_templates (name, page_type, system_prompt, structure) VALUES
('Service Page', 'service', 'Write a comprehensive service page for a home builder. Include benefits, process, and FAQ.', '{"sections": ["hero", "benefits", "process", "testimonials", "faq", "cta"]}'::jsonb),
('Community Page', 'location', 'Write a community/neighborhood page highlighting the area, amenities, and available homes.', '{"sections": ["hero", "about_area", "amenities", "homes", "schools", "cta"]}'::jsonb),
('Blog Post', 'blog', 'Write an informative blog post about home buying, building, or real estate topics.', '{"sections": ["intro", "body", "key_takeaways", "cta"]}'::jsonb);
