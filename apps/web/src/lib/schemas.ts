import { z } from 'zod';

export const businessDetailsSchema = z.object({
  name: z.string().min(2, 'Business name is required'),
  domain: z.string().url('Must be a valid URL').or(z.string().min(3, 'Domain is required')),
  industry: z.string().min(1, 'Industry is required'),
  business_type: z.string().min(1, 'Business type is required'),
  target_locations: z.array(z.string()).min(1, 'At least one target location'),
});

export const goalsSchema = z.object({
  primary_goal: z.string().min(1, 'Primary goal is required'),
  secondary_goal: z.string().optional(),
  kpi_organic_growth: z.number().min(0).max(500).optional(),
  kpi_keywords_top10: z.number().min(0).max(10000).optional(),
  kpi_conversions_growth: z.number().min(0).max(500).optional(),
});

export const brandVoiceSchema = z.object({
  tone: z.string().min(1, 'Tone is required'),
  style: z.string().min(1, 'Style is required'),
  avoid_topics: z.array(z.string()).optional(),
  example_content_url: z.string().url().optional().or(z.literal('')),
});

export const apiAccessSchema = z.object({
  google_oauth: z.boolean().optional(),
  wp_base_url: z.string().url().optional().or(z.literal('')),
  wp_app_user: z.string().optional(),
  wp_app_password: z.string().optional(),
  webflow_api_token: z.string().optional(),
  asana_pat: z.string().optional(),
});

export const onboardingSchema = z.object({
  business: businessDetailsSchema,
  goals: goalsSchema,
  brand: brandVoiceSchema,
  api_access: apiAccessSchema,
});

export type OnboardingData = z.infer<typeof onboardingSchema>;
