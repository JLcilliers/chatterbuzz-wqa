'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

// ── Mock Data ──────────────────────────────────────────────────────────────

const MOCK_CLIENT = {
  id: 'demo-client-001',
  name: 'Acme Plumbing Co.',
  domain: 'https://acmeplumbing.com',
  industry: 'Home Services',
  business_type: 'Local Service',
  target_locations: ['Atlanta, GA', 'Marietta, GA', 'Decatur, GA'],
  goals: {
    primary: 'Increase organic traffic by 40%',
    secondary: 'Rank top 10 for 50 local keywords',
    kpi_targets: { organic_sessions_growth: 40, keyword_rankings_top10: 50, conversions_growth: 25 },
  },
  brand_voice: { tone: 'Professional & Friendly', style: 'Conversational', avoid: ['Slang', 'Technical jargon'] },
  status: 'active',
};

const MOCK_PIPELINE_STEPS = [
  { name: 'Google Search Console — Pull query data', duration: 1800 },
  { name: 'Google Analytics 4 — Pull traffic data', duration: 1400 },
  { name: 'Crawl site — Extract meta, headings, content', duration: 2200 },
  { name: 'WQA Engine — Score all pages', duration: 2000 },
  { name: 'Keyword gap analysis', duration: 1200 },
  { name: 'Content opportunity detection', duration: 1000 },
  { name: 'Generate content briefs', duration: 1600 },
  { name: 'AI content generation (Claude)', duration: 2400 },
  { name: 'Quality gate — Score & filter content', duration: 800 },
  { name: 'Build Excel report', duration: 600 },
  { name: 'Generate monthly report', duration: 500 },
  { name: 'Pipeline complete', duration: 0 },
];

const MOCK_WQA_RESULTS = [
  { url: '/services/drain-cleaning', title: 'Drain Cleaning Services', overall_score: 34, priority: 'high', action: 'Rewrite', word_count: 180, issues: ['Thin content', 'Missing H2s', 'No internal links'] },
  { url: '/services/water-heater', title: 'Water Heater Installation', overall_score: 52, priority: 'high', action: 'Expand', word_count: 420, issues: ['Below word count threshold', 'Weak meta description'] },
  { url: '/about', title: 'About Acme Plumbing', overall_score: 61, priority: 'medium', action: 'Optimize', word_count: 680, issues: ['Missing schema markup', 'No CTA'] },
  { url: '/services/emergency', title: 'Emergency Plumbing 24/7', overall_score: 45, priority: 'high', action: 'Rewrite', word_count: 250, issues: ['Thin content', 'Duplicate title tag', 'Missing alt text'] },
  { url: '/blog/prevent-frozen-pipes', title: 'How to Prevent Frozen Pipes', overall_score: 78, priority: 'low', action: 'Minor tweaks', word_count: 1200, issues: ['Outdated date', 'Broken outbound link'] },
  { url: '/services/sewer-line', title: 'Sewer Line Repair', overall_score: 41, priority: 'high', action: 'Rewrite', word_count: 210, issues: ['Thin content', 'No location mentions', 'Missing FAQ schema'] },
  { url: '/contact', title: 'Contact Us', overall_score: 88, priority: 'low', action: 'None', word_count: 150, issues: [] },
  { url: '/services/leak-detection', title: 'Leak Detection', overall_score: 55, priority: 'medium', action: 'Expand', word_count: 500, issues: ['Below threshold', 'Missing internal links'] },
];

const MOCK_CONTENT_QUEUE = [
  { title: 'Complete Guide to Drain Cleaning in Atlanta', type: 'Page Rewrite', status: 'review', score: 82, word_count: 1450 },
  { title: 'Emergency Plumbing Services — Available 24/7', type: 'Page Rewrite', status: 'review', score: 79, word_count: 1320 },
  { title: 'Sewer Line Repair & Replacement in Atlanta, GA', type: 'Page Rewrite', status: 'approved', score: 85, word_count: 1580 },
  { title: 'Water Heater Installation & Repair Guide', type: 'Page Expansion', status: 'draft', score: 74, word_count: 1100 },
  { title: '10 Signs You Need a Plumber (Don\'t Ignore #7)', type: 'New Blog Post', status: 'review', score: 88, word_count: 1800 },
  { title: 'Tankless vs. Traditional Water Heaters: Which Is Right?', type: 'New Blog Post', status: 'draft', score: 71, word_count: 1650 },
];

const MOCK_KEYWORDS = [
  { keyword: 'plumber atlanta', volume: 2400, position: 18, opportunity: 'high' },
  { keyword: 'emergency plumber near me', volume: 1900, position: 32, opportunity: 'high' },
  { keyword: 'drain cleaning atlanta', volume: 880, position: 8, opportunity: 'medium' },
  { keyword: 'water heater installation atlanta', volume: 720, position: 24, opportunity: 'high' },
  { keyword: 'sewer line repair cost', volume: 590, position: null, opportunity: 'new content' },
  { keyword: 'leak detection service', volume: 480, position: 15, opportunity: 'medium' },
  { keyword: 'tankless water heater atlanta', volume: 390, position: null, opportunity: 'new content' },
  { keyword: 'plumbing company near me', volume: 3200, position: 45, opportunity: 'high' },
];

// ── Types ──────────────────────────────────────────────────────────────────

type DemoPhase = 'intro' | 'onboarding' | 'pipeline' | 'wqa' | 'content' | 'dashboard' | 'portal';

const PHASES: { key: DemoPhase; label: string }[] = [
  { key: 'intro', label: 'Overview' },
  { key: 'onboarding', label: '1. Onboarding' },
  { key: 'pipeline', label: '2. Pipeline' },
  { key: 'wqa', label: '3. WQA Analysis' },
  { key: 'content', label: '4. Content Engine' },
  { key: 'dashboard', label: '5. Ops Dashboard' },
  { key: 'portal', label: '6. Client Portal' },
];

// ── Helpers ────────────────────────────────────────────────────────────────

function Badge({ color, children }: { color: string; children: React.ReactNode }) {
  const colors: Record<string, string> = {
    green: 'bg-green-900/50 text-green-400 border-green-800',
    red: 'bg-red-900/50 text-red-400 border-red-800',
    yellow: 'bg-yellow-900/50 text-yellow-400 border-yellow-800',
    blue: 'bg-blue-900/50 text-blue-400 border-blue-800',
    purple: 'bg-purple-900/50 text-purple-400 border-purple-800',
    gray: 'bg-gray-800 text-gray-400 border-gray-700',
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-xs font-medium ${colors[color] || colors.gray}`}>
      {children}
    </span>
  );
}

function Card({ title, children, className = '' }: { title?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`border border-gray-800 rounded-lg bg-gray-900 ${className}`}>
      {title && <div className="px-5 py-3 border-b border-gray-800 font-semibold text-sm">{title}</div>}
      <div className="p-5">{children}</div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function DemoPage() {
  const [phase, setPhase] = useState<DemoPhase>('intro');
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [onboardingStep, setOnboardingStep] = useState(0);
  const [contentStatuses, setContentStatuses] = useState<string[]>(MOCK_CONTENT_QUEUE.map((c) => c.status));

  // Auto-advance pipeline
  useEffect(() => {
    if (!pipelineRunning || pipelineStep >= MOCK_PIPELINE_STEPS.length - 1) {
      if (pipelineStep >= MOCK_PIPELINE_STEPS.length - 1) setPipelineRunning(false);
      return;
    }
    const timer = setTimeout(() => {
      setPipelineStep((s) => s + 1);
    }, MOCK_PIPELINE_STEPS[pipelineStep + 1]?.duration || 800);
    return () => clearTimeout(timer);
  }, [pipelineRunning, pipelineStep]);

  function startPipeline() {
    setPipelineStep(0);
    setPipelineRunning(true);
  }

  function goTo(p: DemoPhase) {
    setPhase(p);
    if (p === 'pipeline') {
      setPipelineStep(-1);
      setPipelineRunning(false);
    }
  }

  const phaseIdx = PHASES.findIndex((p) => p.key === phase);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold tracking-tight">Chatterbuzz</span>
            <Badge color="purple">Platform Demo</Badge>
          </div>
          <div className="flex gap-1">
            {PHASES.map((p, i) => (
              <button
                key={p.key}
                onClick={() => goTo(p.key)}
                className={`px-3 py-1.5 text-xs rounded transition ${
                  phase === p.key
                    ? 'bg-blue-600 text-white font-medium'
                    : 'text-gray-400 hover:bg-gray-800'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* ── INTRO ─────────────────────────────────────────── */}
        {phase === 'intro' && (
          <div className="space-y-8">
            <div className="text-center py-12">
              <h1 className="text-4xl font-bold mb-4">Chatterbuzz SEO Automation Platform</h1>
              <p className="text-lg text-gray-400 max-w-2xl mx-auto">
                End-to-end SEO pipeline: from client onboarding to automated content generation,
                quality analysis, and publishing — all orchestrated through a single platform.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { step: '1', title: 'Client Onboarding', desc: 'Client fills out business details, goals, brand voice, and API credentials. Saved to Supabase, triggers the first pipeline run.' },
                { step: '2', title: 'Data Pipeline', desc: 'Pulls data from Google Search Console, GA4, crawls the site, runs keyword analysis — all automated via orchestrated workers.' },
                { step: '3', title: 'WQA Analysis', desc: 'Every page scored on content quality, technical SEO, and keyword optimization. Prioritized action items generated.' },
                { step: '4', title: 'Content Engine', desc: 'AI generates optimized content for pages flagged by WQA. Quality gate scores each piece before it enters the review queue.' },
                { step: '5', title: 'Ops Dashboard', desc: 'Internal team monitors all clients, pipeline health, content queue, and alerts from one central dashboard.' },
                { step: '6', title: 'Client Portal', desc: 'Clients review and approve/reject generated content, view reports, and track their SEO performance over time.' },
              ].map((item) => (
                <Card key={item.step}>
                  <div className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm font-bold">
                      {item.step}
                    </span>
                    <div>
                      <h3 className="font-semibold mb-1">{item.title}</h3>
                      <p className="text-sm text-gray-400">{item.desc}</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>

            <div className="text-center pt-4 flex gap-4 justify-center">
              <button
                onClick={() => goTo('onboarding')}
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium text-lg transition"
              >
                Start Walkthrough
              </button>
              <Link
                href="/demo/autoplay"
                className="px-8 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg font-medium text-lg transition inline-flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.84A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.27l9.344-5.891a1.5 1.5 0 000-2.538L6.3 2.841z" />
                </svg>
                Autoplay Demo
              </Link>
            </div>

            <Card title="Tech Stack">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                {[
                  { label: 'Frontend', value: 'Next.js 14 (2 apps)' },
                  { label: 'Backend', value: 'FastAPI Gateway' },
                  { label: 'Database', value: 'Supabase (18 tables)' },
                  { label: 'AI', value: 'Claude API' },
                  { label: 'Pipeline', value: 'Python Workers + n8n' },
                  { label: 'Hosting', value: 'Vercel + Railway' },
                  { label: 'Monorepo', value: 'pnpm + Turborepo' },
                  { label: 'Automation', value: '10 n8n workflows' },
                ].map((item) => (
                  <div key={item.label}>
                    <p className="text-gray-500">{item.label}</p>
                    <p className="font-medium">{item.value}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ── ONBOARDING ────────────────────────────────────── */}
        {phase === 'onboarding' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Step 1: Client Onboarding</h2>
            <p className="text-gray-400">New clients complete a multi-step wizard. Data is validated and saved to Supabase.</p>

            <div className="flex gap-2 mb-4">
              {['Business Details', 'Goals & KPIs', 'Brand Voice', 'API Access'].map((label, i) => (
                <button
                  key={label}
                  onClick={() => setOnboardingStep(i)}
                  className={`px-4 py-2 text-sm rounded transition ${
                    onboardingStep === i ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            <Card>
              {onboardingStep === 0 && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Business Name</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.name}</div>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Website</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.domain}</div>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Industry</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.industry}</div>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Business Type</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.business_type}</div>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">Target Locations</label>
                    <div className="flex gap-2">
                      {MOCK_CLIENT.target_locations.map((loc) => (
                        <Badge key={loc} color="blue">{loc}</Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {onboardingStep === 1 && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">Primary Goal</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.goals.primary}</div>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">Secondary Goal</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.goals.secondary}</div>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-2">KPI Targets</label>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-800 border border-gray-700 rounded p-3 text-center">
                        <p className="text-2xl font-bold text-green-400">+{MOCK_CLIENT.goals.kpi_targets.organic_sessions_growth}%</p>
                        <p className="text-xs text-gray-400 mt-1">Organic Traffic</p>
                      </div>
                      <div className="bg-gray-800 border border-gray-700 rounded p-3 text-center">
                        <p className="text-2xl font-bold text-blue-400">{MOCK_CLIENT.goals.kpi_targets.keyword_rankings_top10}</p>
                        <p className="text-xs text-gray-400 mt-1">Keywords in Top 10</p>
                      </div>
                      <div className="bg-gray-800 border border-gray-700 rounded p-3 text-center">
                        <p className="text-2xl font-bold text-purple-400">+{MOCK_CLIENT.goals.kpi_targets.conversions_growth}%</p>
                        <p className="text-xs text-gray-400 mt-1">Conversions</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              {onboardingStep === 2 && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Tone</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.brand_voice.tone}</div>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Style</label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.brand_voice.style}</div>
                  </div>
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">Topics to Avoid</label>
                    <div className="flex gap-2">
                      {MOCK_CLIENT.brand_voice.avoid.map((t) => (
                        <Badge key={t} color="red">{t}</Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {onboardingStep === 3 && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Google OAuth</label>
                      <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-400"></span> Connected
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">WordPress</label>
                      <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-400"></span> acmeplumbing.com/wp-json
                      </div>
                    </div>
                  </div>
                  <div className="bg-blue-900/20 border border-blue-800 rounded p-3 text-sm text-blue-300">
                    Credentials are encrypted via pgcrypto and stored in the client_api_credentials table. Only the pipeline workers can decrypt them at runtime.
                  </div>
                </div>
              )}
            </Card>

            <div className="flex justify-between items-center">
              <button
                onClick={() => setOnboardingStep(Math.max(0, onboardingStep - 1))}
                disabled={onboardingStep === 0}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm disabled:opacity-30"
              >
                Previous
              </button>
              {onboardingStep < 3 ? (
                <button
                  onClick={() => setOnboardingStep(onboardingStep + 1)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium"
                >
                  Next Step
                </button>
              ) : (
                <button
                  onClick={() => goTo('pipeline')}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium"
                >
                  Submit &amp; Start Pipeline
                </button>
              )}
            </div>

            <Card title="What happens on submit">
              <ol className="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                <li>Zod validates all input fields</li>
                <li>Client record created in Supabase <code className="text-gray-300">clients</code> table</li>
                <li>API credentials encrypted and stored in <code className="text-gray-300">client_api_credentials</code></li>
                <li>Default business rules inserted (min 800 words, 100 posts/week limit)</li>
                <li>Initial pipeline run created with status <code className="text-gray-300">pending</code></li>
                <li>Client status updated to <code className="text-gray-300">active</code></li>
                <li>POST to FastAPI gateway triggers the pipeline</li>
              </ol>
            </Card>
          </div>
        )}

        {/* ── PIPELINE ──────────────────────────────────────── */}
        {phase === 'pipeline' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Step 2: Data Pipeline</h2>
                <p className="text-gray-400 mt-1">Orchestrated workers pull data from all sources, analyze, and generate outputs.</p>
              </div>
              {!pipelineRunning && pipelineStep < MOCK_PIPELINE_STEPS.length - 1 && (
                <button
                  onClick={startPipeline}
                  className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium transition"
                >
                  Run Pipeline
                </button>
              )}
              {pipelineStep >= MOCK_PIPELINE_STEPS.length - 1 && (
                <Badge color="green">Complete</Badge>
              )}
            </div>

            <Card title={`Pipeline: ${MOCK_CLIENT.name}`}>
              <div className="space-y-3">
                {MOCK_PIPELINE_STEPS.map((step, i) => {
                  const isDone = i < pipelineStep;
                  const isActive = i === pipelineStep && pipelineRunning;
                  const isPending = i > pipelineStep;

                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-3 px-3 py-2 rounded transition-all duration-300 ${
                        isActive ? 'bg-blue-900/30 border border-blue-800' :
                        isDone ? 'bg-gray-800/50' : ''
                      }`}
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                        isDone ? 'bg-green-600 text-white' :
                        isActive ? 'bg-blue-600 text-white animate-pulse' :
                        'bg-gray-800 text-gray-500'
                      }`}>
                        {isDone ? '\u2713' : i + 1}
                      </div>
                      <span className={`text-sm ${isPending ? 'text-gray-600' : isDone ? 'text-gray-400' : 'text-white font-medium'}`}>
                        {step.name}
                      </span>
                      {isActive && (
                        <span className="ml-auto text-xs text-blue-400 animate-pulse">Processing...</span>
                      )}
                      {isDone && (
                        <span className="ml-auto text-xs text-gray-600">{(step.duration / 1000).toFixed(1)}s</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </Card>

            {pipelineStep >= MOCK_PIPELINE_STEPS.length - 1 && (
              <div className="flex justify-end">
                <button
                  onClick={() => goTo('wqa')}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition"
                >
                  View WQA Results
                </button>
              </div>
            )}

            <Card title="Architecture">
              <div className="text-sm text-gray-400 space-y-2">
                <p>The pipeline orchestrator dispatches work to specialized Python workers:</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                  {['GSC Worker', 'GA4 Worker', 'Crawler Worker', 'WQA Engine', 'Keyword Analyzer', 'Content Detector', 'Content Generator', 'Excel Writer'].map((w) => (
                    <div key={w} className="bg-gray-800 rounded px-3 py-2 text-center text-xs font-medium text-gray-300">{w}</div>
                  ))}
                </div>
                <p className="mt-3">Each worker reads/writes to Supabase. The orchestrator tracks progress in the <code className="text-gray-300">pipeline_runs</code> table with real-time status updates.</p>
              </div>
            </Card>
          </div>
        )}

        {/* ── WQA ANALYSIS ──────────────────────────────────── */}
        {phase === 'wqa' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Step 3: WQA Analysis</h2>
            <p className="text-gray-400">Every page on the site is scored and prioritized. The WQA engine identifies what needs fixing and why.</p>

            <div className="grid grid-cols-4 gap-4">
              <Card>
                <p className="text-sm text-gray-400">Pages Analyzed</p>
                <p className="text-3xl font-bold mt-1">{MOCK_WQA_RESULTS.length}</p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">High Priority</p>
                <p className="text-3xl font-bold mt-1 text-red-400">{MOCK_WQA_RESULTS.filter((r) => r.priority === 'high').length}</p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">Avg Score</p>
                <p className="text-3xl font-bold mt-1 text-yellow-400">
                  {Math.round(MOCK_WQA_RESULTS.reduce((s, r) => s + r.overall_score, 0) / MOCK_WQA_RESULTS.length)}
                </p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">Content Rewrites Needed</p>
                <p className="text-3xl font-bold mt-1 text-blue-400">{MOCK_WQA_RESULTS.filter((r) => r.action === 'Rewrite').length}</p>
              </Card>
            </div>

            <Card title="Page-Level Results">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800">
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">URL</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Score</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Priority</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Action</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Words</th>
                      <th className="text-left px-3 py-2 text-gray-400 font-medium">Issues</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {MOCK_WQA_RESULTS.map((result) => (
                      <tr key={result.url}>
                        <td className="px-3 py-2 text-blue-400">{result.url}</td>
                        <td className="px-3 py-2">
                          <span className={`font-bold ${
                            result.overall_score >= 70 ? 'text-green-400' :
                            result.overall_score >= 50 ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {result.overall_score}
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          <Badge color={result.priority === 'high' ? 'red' : result.priority === 'medium' ? 'yellow' : 'green'}>
                            {result.priority}
                          </Badge>
                        </td>
                        <td className="px-3 py-2">{result.action}</td>
                        <td className="px-3 py-2 text-gray-400">{result.word_count}</td>
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {result.issues.map((issue) => (
                              <span key={issue} className="text-xs bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">{issue}</span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Keyword Opportunities">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Keyword</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Monthly Volume</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Current Position</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Opportunity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {MOCK_KEYWORDS.map((kw) => (
                    <tr key={kw.keyword}>
                      <td className="px-3 py-2 font-medium">{kw.keyword}</td>
                      <td className="px-3 py-2 text-gray-300">{kw.volume.toLocaleString()}</td>
                      <td className="px-3 py-2 text-gray-400">{kw.position ?? 'Not ranking'}</td>
                      <td className="px-3 py-2">
                        <Badge color={kw.opportunity === 'high' ? 'red' : kw.opportunity === 'new content' ? 'purple' : 'yellow'}>
                          {kw.opportunity}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <div className="flex justify-end">
              <button onClick={() => goTo('content')} className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition">
                View Content Engine
              </button>
            </div>
          </div>
        )}

        {/* ── CONTENT ENGINE ────────────────────────────────── */}
        {phase === 'content' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Step 4: Content Engine</h2>
            <p className="text-gray-400">AI generates optimized content for flagged pages. Each piece passes through a quality gate (threshold: 70/100) before entering the review queue.</p>

            <div className="grid grid-cols-4 gap-4">
              <Card>
                <p className="text-sm text-gray-400">Generated</p>
                <p className="text-3xl font-bold mt-1">{MOCK_CONTENT_QUEUE.length}</p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">Awaiting Review</p>
                <p className="text-3xl font-bold mt-1 text-yellow-400">{contentStatuses.filter((s) => s === 'review').length}</p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">Approved</p>
                <p className="text-3xl font-bold mt-1 text-green-400">{contentStatuses.filter((s) => s === 'approved').length}</p>
              </Card>
              <Card>
                <p className="text-sm text-gray-400">Avg Quality Score</p>
                <p className="text-3xl font-bold mt-1 text-blue-400">
                  {Math.round(MOCK_CONTENT_QUEUE.reduce((s, c) => s + c.score, 0) / MOCK_CONTENT_QUEUE.length)}
                </p>
              </Card>
            </div>

            <Card title="Content Queue">
              <div className="space-y-3">
                {MOCK_CONTENT_QUEUE.map((item, i) => (
                  <div key={i} className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3">
                    <div className="flex-1">
                      <p className="font-medium text-sm">{item.title}</p>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                        <span>{item.type}</span>
                        <span>{item.word_count} words</span>
                        <span>Quality: <span className={item.score >= 80 ? 'text-green-400' : item.score >= 70 ? 'text-yellow-400' : 'text-red-400'}>{item.score}/100</span></span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge color={contentStatuses[i] === 'approved' ? 'green' : contentStatuses[i] === 'rejected' ? 'red' : contentStatuses[i] === 'review' ? 'yellow' : 'gray'}>
                        {contentStatuses[i]}
                      </Badge>
                      {contentStatuses[i] === 'review' && (
                        <div className="flex gap-1 ml-2">
                          <button
                            onClick={() => setContentStatuses((prev) => prev.map((s, j) => j === i ? 'approved' : s))}
                            className="px-2 py-1 bg-green-700 hover:bg-green-600 rounded text-xs"
                          >
                            Approve
                          </button>
                          <button
                            onClick={() => setContentStatuses((prev) => prev.map((s, j) => j === i ? 'rejected' : s))}
                            className="px-2 py-1 bg-red-700 hover:bg-red-600 rounded text-xs"
                          >
                            Reject
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Quality Gate">
              <div className="text-sm text-gray-400 space-y-2">
                <p>Every piece of AI-generated content is scored across 4 dimensions (25 points each):</p>
                <div className="grid grid-cols-4 gap-3 mt-3">
                  {[
                    { dim: 'Relevance', desc: 'Matches target keyword & intent' },
                    { dim: 'Depth', desc: 'Comprehensive coverage of topic' },
                    { dim: 'Readability', desc: 'Clear, well-structured prose' },
                    { dim: 'SEO', desc: 'Headings, links, meta, schema' },
                  ].map((d) => (
                    <div key={d.dim} className="bg-gray-800 rounded p-3">
                      <p className="font-medium text-gray-200">{d.dim}</p>
                      <p className="text-xs text-gray-500 mt-1">{d.desc}</p>
                      <p className="text-lg font-bold text-blue-400 mt-2">/25</p>
                    </div>
                  ))}
                </div>
                <p className="mt-3">Minimum score to pass: <span className="text-white font-bold">70/100</span>. Content below threshold is regenerated or flagged for manual review.</p>
              </div>
            </Card>

            <div className="flex justify-end">
              <button onClick={() => goTo('dashboard')} className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition">
                View Ops Dashboard
              </button>
            </div>
          </div>
        )}

        {/* ── OPS DASHBOARD ─────────────────────────────────── */}
        {phase === 'dashboard' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Step 5: Ops Dashboard</h2>
            <p className="text-gray-400">Internal team&apos;s command center. Monitor all clients, pipeline health, and content across the platform.</p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: 'Active Clients', value: '12', color: 'text-white' },
                { label: 'Pipeline Runs (24h)', value: '34', color: 'text-blue-400' },
                { label: 'Content Pending Review', value: '8', color: 'text-yellow-400' },
                { label: 'Active Alerts', value: '2', color: 'text-red-400' },
              ].map((card) => (
                <Card key={card.label}>
                  <p className="text-sm text-gray-400">{card.label}</p>
                  <p className={`text-3xl font-bold mt-1 ${card.color}`}>{card.value}</p>
                </Card>
              ))}
            </div>

            <Card title="Client Overview">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Client</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Domain</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Status</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Last Pipeline</th>
                    <th className="text-left px-3 py-2 text-gray-400 font-medium">Content Queue</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {[
                    { name: 'Acme Plumbing Co.', domain: 'acmeplumbing.com', status: 'active', lastRun: '2h ago', queue: 6 },
                    { name: 'Sunset Dental', domain: 'sunsetdental.com', status: 'active', lastRun: '5h ago', queue: 3 },
                    { name: 'Peak Fitness ATL', domain: 'peakfitnessatl.com', status: 'active', lastRun: '1d ago', queue: 0 },
                    { name: 'GreenLeaf Landscaping', domain: 'greenleaflandscape.com', status: 'onboarding', lastRun: '—', queue: 0 },
                  ].map((c) => (
                    <tr key={c.name}>
                      <td className="px-3 py-2 text-blue-400 font-medium">{c.name}</td>
                      <td className="px-3 py-2 text-gray-300">{c.domain}</td>
                      <td className="px-3 py-2"><Badge color={c.status === 'active' ? 'green' : 'yellow'}>{c.status}</Badge></td>
                      <td className="px-3 py-2 text-gray-400">{c.lastRun}</td>
                      <td className="px-3 py-2">{c.queue > 0 ? <Badge color="yellow">{c.queue} pending</Badge> : <span className="text-gray-600">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <div className="grid grid-cols-2 gap-4">
              <Card title="Recent Pipeline Runs">
                <div className="space-y-2">
                  {[
                    { client: 'Acme Plumbing', type: 'full_pipeline', status: 'completed', time: '2h ago' },
                    { client: 'Sunset Dental', type: 'content_refresh', status: 'completed', time: '5h ago' },
                    { client: 'Peak Fitness', type: 'full_pipeline', status: 'completed', time: '1d ago' },
                    { client: 'Acme Plumbing', type: 'index_monitor', status: 'failed', time: '3h ago' },
                  ].map((run, i) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <div>
                        <span className="text-gray-300">{run.client}</span>
                        <span className="text-gray-600 ml-2">{run.type}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge color={run.status === 'completed' ? 'green' : 'red'}>{run.status}</Badge>
                        <span className="text-xs text-gray-600">{run.time}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              <Card title="Alerts">
                <div className="space-y-3">
                  <div className="flex items-start gap-3 bg-red-900/20 border border-red-900 rounded p-3">
                    <span className="text-red-400 text-lg">!</span>
                    <div>
                      <p className="text-sm font-medium text-red-300">Pipeline Failed: Acme Plumbing</p>
                      <p className="text-xs text-gray-400 mt-1">Index monitor timed out — GSC API rate limit exceeded. Auto-retry scheduled in 1h.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 bg-yellow-900/20 border border-yellow-900 rounded p-3">
                    <span className="text-yellow-400 text-lg">!</span>
                    <div>
                      <p className="text-sm font-medium text-yellow-300">Content Below Threshold</p>
                      <p className="text-xs text-gray-400 mt-1">1 content piece for Sunset Dental scored 62/100. Queued for regeneration.</p>
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            <div className="flex justify-end">
              <button onClick={() => goTo('portal')} className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition">
                View Client Portal
              </button>
            </div>
          </div>
        )}

        {/* ── CLIENT PORTAL ─────────────────────────────────── */}
        {phase === 'portal' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Step 6: Client Portal</h2>
            <p className="text-gray-400">White-label portal where clients review content, approve publications, and track performance.</p>

            <div className="bg-white text-gray-900 rounded-xl overflow-hidden border shadow-lg">
              <div className="bg-gray-50 border-b px-6 py-4 flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-lg">{MOCK_CLIENT.name}</h3>
                  <p className="text-sm text-gray-500">{MOCK_CLIENT.domain}</p>
                </div>
                <Badge color="green">Active</Badge>
              </div>

              <div className="p-6 space-y-6">
                <div className="grid grid-cols-3 gap-4">
                  <div className="border rounded-lg p-4 text-center">
                    <p className="text-3xl font-bold text-green-600">+32%</p>
                    <p className="text-sm text-gray-500 mt-1">Organic Traffic (MoM)</p>
                  </div>
                  <div className="border rounded-lg p-4 text-center">
                    <p className="text-3xl font-bold text-blue-600">18</p>
                    <p className="text-sm text-gray-500 mt-1">Keywords in Top 10</p>
                  </div>
                  <div className="border rounded-lg p-4 text-center">
                    <p className="text-3xl font-bold text-purple-600">6</p>
                    <p className="text-sm text-gray-500 mt-1">Content Pieces Ready</p>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Content Awaiting Your Review</h4>
                  <div className="space-y-3">
                    {MOCK_CONTENT_QUEUE.filter((c) => c.status === 'review').map((item, i) => (
                      <div key={i} className="border rounded-lg p-4 flex items-center justify-between">
                        <div>
                          <p className="font-medium">{item.title}</p>
                          <p className="text-sm text-gray-500">{item.type} &middot; {item.word_count} words &middot; Quality: {item.score}/100</p>
                        </div>
                        <div className="flex gap-2">
                          <button className="px-3 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700">Approve</button>
                          <button className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300">Reject</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Monthly Report — February 2026</h4>
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Pages Optimized</p>
                        <p className="font-bold text-lg">5</p>
                      </div>
                      <div>
                        <p className="text-gray-500">New Content Published</p>
                        <p className="font-bold text-lg">3</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Avg. WQA Score</p>
                        <p className="font-bold text-lg">72 (+18)</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Impressions</p>
                        <p className="font-bold text-lg">24.3K</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="text-center pt-8 pb-4">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8">
                <h3 className="text-2xl font-bold mb-2">End-to-End Demo Complete</h3>
                <p className="text-blue-100 mb-6">
                  From onboarding to content delivery — fully automated, quality-gated, client-facing.
                </p>
                <button
                  onClick={() => goTo('intro')}
                  className="px-6 py-2 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-100 transition"
                >
                  Restart Demo
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Phase Navigation ──────────────────────────────── */}
        <div className="flex justify-between items-center mt-12 pt-6 border-t border-gray-800">
          {phaseIdx > 0 ? (
            <button
              onClick={() => goTo(PHASES[phaseIdx - 1].key)}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white transition"
            >
              &larr; {PHASES[phaseIdx - 1].label}
            </button>
          ) : <div />}
          {phaseIdx < PHASES.length - 1 ? (
            <button
              onClick={() => goTo(PHASES[phaseIdx + 1].key)}
              className="px-4 py-2 text-sm text-gray-400 hover:text-white transition"
            >
              {PHASES[phaseIdx + 1].label} &rarr;
            </button>
          ) : <div />}
        </div>
      </div>
    </main>
  );
}
