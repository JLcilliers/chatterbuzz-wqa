'use client';

import { useState } from 'react';
import type { OnboardingData } from '@/lib/schemas';

const STEPS = [
  'Business Details',
  'Goals & KPIs',
  'Brand & Voice',
  'API Access',
  'Review & Submit',
] as const;

const INDUSTRIES = [
  'Real Estate', 'Home Services', 'Healthcare', 'Legal', 'Financial Services',
  'E-commerce', 'SaaS', 'Education', 'Hospitality', 'Manufacturing', 'Other',
];

const TONES = ['Professional', 'Friendly', 'Authoritative', 'Casual', 'Technical'];
const STYLES = ['Informative', 'Persuasive', 'Conversational', 'Data-driven', 'Storytelling'];

export default function OnboardPage() {
  const [step, setStep] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const [data, setData] = useState<OnboardingData>({
    business: { name: '', domain: '', industry: '', business_type: '', target_locations: [''] },
    goals: { primary_goal: '', secondary_goal: '' },
    brand: { tone: '', style: '', avoid_topics: [], example_content_url: '' },
    api_access: { google_oauth: false, wp_base_url: '', wp_app_user: '', wp_app_password: '', webflow_api_token: '', asana_pat: '' },
  });

  function updateBusiness(field: string, value: string | string[]) {
    setData(d => ({ ...d, business: { ...d.business, [field]: value } }));
  }

  function updateGoals(field: string, value: string | number) {
    setData(d => ({ ...d, goals: { ...d.goals, [field]: value } }));
  }

  function updateBrand(field: string, value: string | string[]) {
    setData(d => ({ ...d, brand: { ...d.brand, [field]: value } }));
  }

  function updateApi(field: string, value: string | boolean) {
    setData(d => ({ ...d, api_access: { ...d.api_access, [field]: value } }));
  }

  async function handleSubmit() {
    setSubmitting(true);
    setError('');
    try {
      const res = await fetch('/api/onboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Onboarding failed');
      window.location.href = `/portal/${result.clientId}`;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unexpected error');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="max-w-2xl mx-auto py-12 px-4">
      <h1 className="text-2xl font-bold mb-8">Client Onboarding</h1>

      {/* Step indicators */}
      <div className="flex gap-2 mb-8">
        {STEPS.map((label, i) => (
          <div
            key={label}
            className={`flex-1 text-center text-sm py-2 rounded ${
              i === step
                ? 'bg-blue-600 text-white'
                : i < step
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-200 text-gray-600'
            }`}
          >
            {label}
          </div>
        ))}
      </div>

      {/* Step content */}
      <div className="border rounded-lg p-8 mb-6 bg-white">
        {step === 0 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Business Details</h2>
            <input
              placeholder="Business Name"
              value={data.business.name}
              onChange={e => updateBusiness('name', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
            <input
              placeholder="Domain (e.g. example.com)"
              value={data.business.domain}
              onChange={e => updateBusiness('domain', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
            <select
              value={data.business.industry}
              onChange={e => updateBusiness('industry', e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select Industry</option>
              {INDUSTRIES.map(i => <option key={i} value={i}>{i}</option>)}
            </select>
            <input
              placeholder="Business Type (e.g. Home Builder, Law Firm)"
              value={data.business.business_type}
              onChange={e => updateBusiness('business_type', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
            <div>
              <label className="block text-sm font-medium mb-1">Target Locations</label>
              {data.business.target_locations.map((loc, i) => (
                <div key={i} className="flex gap-2 mb-2">
                  <input
                    placeholder="City, State"
                    value={loc}
                    onChange={e => {
                      const locs = [...data.business.target_locations];
                      locs[i] = e.target.value;
                      updateBusiness('target_locations', locs);
                    }}
                    className="flex-1 border rounded px-3 py-2"
                  />
                  {i > 0 && (
                    <button
                      onClick={() => updateBusiness('target_locations', data.business.target_locations.filter((_, j) => j !== i))}
                      className="text-red-500 px-2"
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
              <button
                onClick={() => updateBusiness('target_locations', [...data.business.target_locations, ''])}
                className="text-blue-600 text-sm"
              >
                + Add Location
              </button>
            </div>
          </div>
        )}

        {step === 1 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Goals & KPIs</h2>
            <input
              placeholder="Primary Goal (e.g. Increase organic traffic)"
              value={data.goals.primary_goal}
              onChange={e => updateGoals('primary_goal', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
            <input
              placeholder="Secondary Goal (optional)"
              value={data.goals.secondary_goal || ''}
              onChange={e => updateGoals('secondary_goal', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Organic Growth %</label>
                <input
                  type="number"
                  placeholder="20"
                  value={data.goals.kpi_organic_growth || ''}
                  onChange={e => updateGoals('kpi_organic_growth', Number(e.target.value))}
                  className="w-full border rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Keywords Top 10</label>
                <input
                  type="number"
                  placeholder="50"
                  value={data.goals.kpi_keywords_top10 || ''}
                  onChange={e => updateGoals('kpi_keywords_top10', Number(e.target.value))}
                  className="w-full border rounded px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Conversions Growth %</label>
                <input
                  type="number"
                  placeholder="15"
                  value={data.goals.kpi_conversions_growth || ''}
                  onChange={e => updateGoals('kpi_conversions_growth', Number(e.target.value))}
                  className="w-full border rounded px-3 py-2"
                />
              </div>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Brand & Voice</h2>
            <select
              value={data.brand.tone}
              onChange={e => updateBrand('tone', e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select Tone</option>
              {TONES.map(t => <option key={t} value={t.toLowerCase()}>{t}</option>)}
            </select>
            <select
              value={data.brand.style}
              onChange={e => updateBrand('style', e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              <option value="">Select Style</option>
              {STYLES.map(s => <option key={s} value={s.toLowerCase()}>{s}</option>)}
            </select>
            <div>
              <label className="block text-sm font-medium mb-1">Topics to Avoid (comma-separated)</label>
              <input
                placeholder="jargon, aggressive sales language"
                value={(data.brand.avoid_topics || []).join(', ')}
                onChange={e => updateBrand('avoid_topics', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                className="w-full border rounded px-3 py-2"
              />
            </div>
            <input
              placeholder="Example content URL (optional)"
              value={data.brand.example_content_url || ''}
              onChange={e => updateBrand('example_content_url', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">API Access</h2>
            <div className="p-4 bg-blue-50 rounded-lg">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={data.api_access.google_oauth || false}
                  onChange={e => updateApi('google_oauth', e.target.checked)}
                />
                <span className="font-medium">Connect Google (GA4 + GSC)</span>
              </label>
              <p className="text-sm text-gray-500 mt-1">OAuth flow will start after onboarding</p>
            </div>
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">WordPress (optional)</h3>
              <input
                placeholder="WP REST URL (e.g. https://site.com/wp-json/wp/v2)"
                value={data.api_access.wp_base_url || ''}
                onChange={e => updateApi('wp_base_url', e.target.value)}
                className="w-full border rounded px-3 py-2 mb-2"
              />
              <div className="grid grid-cols-2 gap-2">
                <input
                  placeholder="App Username"
                  value={data.api_access.wp_app_user || ''}
                  onChange={e => updateApi('wp_app_user', e.target.value)}
                  className="border rounded px-3 py-2"
                />
                <input
                  type="password"
                  placeholder="App Password"
                  value={data.api_access.wp_app_password || ''}
                  onChange={e => updateApi('wp_app_password', e.target.value)}
                  className="border rounded px-3 py-2"
                />
              </div>
            </div>
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Webflow (optional)</h3>
              <input
                type="password"
                placeholder="Webflow API Token"
                value={data.api_access.webflow_api_token || ''}
                onChange={e => updateApi('webflow_api_token', e.target.value)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Asana (optional)</h3>
              <input
                type="password"
                placeholder="Asana Personal Access Token"
                value={data.api_access.asana_pat || ''}
                onChange={e => updateApi('asana_pat', e.target.value)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Review & Submit</h2>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-medium">Business</p>
                <p>{data.business.name} — {data.business.domain}</p>
                <p>{data.business.industry} / {data.business.business_type}</p>
                <p>Locations: {data.business.target_locations.filter(Boolean).join(', ')}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-medium">Goals</p>
                <p>Primary: {data.goals.primary_goal}</p>
                {data.goals.secondary_goal && <p>Secondary: {data.goals.secondary_goal}</p>}
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-medium">Brand Voice</p>
                <p>Tone: {data.brand.tone}, Style: {data.brand.style}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded">
                <p className="font-medium">Integrations</p>
                <p>Google: {data.api_access.google_oauth ? 'Yes' : 'No'}</p>
                <p>WordPress: {data.api_access.wp_base_url ? 'Yes' : 'No'}</p>
                <p>Webflow: {data.api_access.webflow_api_token ? 'Yes' : 'No'}</p>
                <p>Asana: {data.api_access.asana_pat ? 'Yes' : 'No'}</p>
              </div>
            </div>
            {error && <p className="text-red-600 text-sm">{error}</p>}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-4 py-2 bg-gray-200 rounded disabled:opacity-50"
        >
          Back
        </button>
        {step === STEPS.length - 1 ? (
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {submitting ? 'Submitting...' : 'Submit & Start'}
          </button>
        ) : (
          <button
            onClick={() => setStep(step + 1)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Next
          </button>
        )}
      </div>
    </main>
  );
}
