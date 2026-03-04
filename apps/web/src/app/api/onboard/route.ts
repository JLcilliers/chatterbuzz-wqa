import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { onboardingSchema } from '@/lib/schemas';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // Validate with zod
    const parsed = onboardingSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: 'Validation failed', details: parsed.error.flatten() },
        { status: 400 }
      );
    }
    const { business, goals, brand, api_access } = parsed.data;

    // Create Supabase service client
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      { auth: { autoRefreshToken: false, persistSession: false } }
    );

    // Insert client
    const { data: client, error: clientError } = await supabase
      .from('clients')
      .insert({
        name: business.name,
        domain: business.domain,
        industry: business.industry,
        business_type: business.business_type,
        target_locations: business.target_locations.filter(Boolean),
        goals: {
          primary: goals.primary_goal,
          secondary: goals.secondary_goal,
          kpi_targets: {
            organic_sessions_growth: goals.kpi_organic_growth,
            keyword_rankings_top10: goals.kpi_keywords_top10,
            conversions_growth: goals.kpi_conversions_growth,
          },
        },
        brand_voice: {
          tone: brand.tone,
          style: brand.style,
          avoid: brand.avoid_topics,
          example_url: brand.example_content_url,
        },
        status: 'onboarding',
      })
      .select('id')
      .single();

    if (clientError || !client) {
      return NextResponse.json({ error: clientError?.message || 'Failed to create client' }, { status: 500 });
    }

    const clientId = client.id;

    // Store API credentials (encrypted via pgcrypto at DB level)
    const credentials: Array<{ client_id: string; provider: string; credentials_encrypted: string; scopes: string[] }> = [];

    if (api_access.wp_base_url && api_access.wp_app_user) {
      credentials.push({
        client_id: clientId,
        provider: 'wordpress',
        credentials_encrypted: JSON.stringify({
          base_url: api_access.wp_base_url,
          username: api_access.wp_app_user,
          password: api_access.wp_app_password,
        }),
        scopes: ['posts', 'pages', 'media'],
      });
    }

    if (api_access.webflow_api_token) {
      credentials.push({
        client_id: clientId,
        provider: 'webflow',
        credentials_encrypted: JSON.stringify({ api_token: api_access.webflow_api_token }),
        scopes: ['cms:read', 'cms:write'],
      });
    }

    if (api_access.asana_pat) {
      credentials.push({
        client_id: clientId,
        provider: 'asana',
        credentials_encrypted: JSON.stringify({ pat: api_access.asana_pat }),
        scopes: ['default'],
      });
    }

    if (credentials.length > 0) {
      await supabase.from('client_api_credentials').insert(credentials);
    }

    // Insert business rules
    const rules = [
      { client_id: clientId, rule_type: 'content_length_minimum', rule_config: { min_words: 800 } },
      { client_id: clientId, rule_type: 'publish_rate_limit', rule_config: { max_per_week: 100 } },
    ];
    await supabase.from('business_rules').insert(rules);

    // Create initial pipeline run (will be picked up by orchestrator)
    await supabase.from('pipeline_runs').insert({
      client_id: clientId,
      pipeline_type: 'initial_setup',
      status: 'pending',
      metadata: { triggered_by: 'onboarding' },
    });

    // Update client status to active
    await supabase.from('clients').update({ status: 'active' }).eq('id', clientId);

    return NextResponse.json({ success: true, clientId });
  } catch (error) {
    console.error('Onboarding error:', error);
    return NextResponse.json({ error: 'Onboarding failed' }, { status: 500 });
  }
}
