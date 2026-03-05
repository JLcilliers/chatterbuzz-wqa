'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';

// ── Mock Data (same as main demo) ─────────────────────────────────────────

const MOCK_CLIENT = {
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
};

const PIPELINE_STEPS = [
  'Google Search Console — Pull query data',
  'Google Analytics 4 — Pull traffic data',
  'Crawl site — Extract meta, headings, content',
  'WQA Engine — Score all pages',
  'Keyword gap analysis',
  'Content opportunity detection',
  'Generate content briefs',
  'AI content generation (Claude)',
  'Quality gate — Score & filter content',
  'Build Excel report',
  'Generate monthly report',
  'Pipeline complete',
];

const WQA_RESULTS = [
  { url: '/services/drain-cleaning', score: 34, priority: 'high', action: 'Rewrite', issues: ['Thin content', 'Missing H2s'] },
  { url: '/services/water-heater', score: 52, priority: 'high', action: 'Expand', issues: ['Below word count'] },
  { url: '/about', score: 61, priority: 'medium', action: 'Optimize', issues: ['Missing schema'] },
  { url: '/services/emergency', score: 45, priority: 'high', action: 'Rewrite', issues: ['Thin content', 'Duplicate title'] },
  { url: '/blog/prevent-frozen-pipes', score: 78, priority: 'low', action: 'Minor tweaks', issues: ['Outdated date'] },
  { url: '/services/sewer-line', score: 41, priority: 'high', action: 'Rewrite', issues: ['Thin content', 'No location mentions'] },
  { url: '/contact', score: 88, priority: 'low', action: 'None', issues: [] },
  { url: '/services/leak-detection', score: 55, priority: 'medium', action: 'Expand', issues: ['Missing internal links'] },
];

const CONTENT_ITEMS = [
  { title: 'Complete Guide to Drain Cleaning in Atlanta', type: 'Page Rewrite', score: 82, words: 1450 },
  { title: 'Emergency Plumbing Services — Available 24/7', type: 'Page Rewrite', score: 79, words: 1320 },
  { title: 'Sewer Line Repair & Replacement in Atlanta, GA', type: 'Page Rewrite', score: 85, words: 1580 },
  { title: 'Water Heater Installation & Repair Guide', type: 'Page Expansion', score: 74, words: 1100 },
  { title: "10 Signs You Need a Plumber (Don't Ignore #7)", type: 'New Blog Post', score: 88, words: 1800 },
];

const KEYWORDS = [
  { keyword: 'plumber atlanta', volume: 2400, position: 18, opportunity: 'high' },
  { keyword: 'emergency plumber near me', volume: 1900, position: 32, opportunity: 'high' },
  { keyword: 'drain cleaning atlanta', volume: 880, position: 8, opportunity: 'medium' },
  { keyword: 'water heater installation atlanta', volume: 720, position: 24, opportunity: 'high' },
  { keyword: 'sewer line repair cost', volume: 590, position: null as number | null, opportunity: 'new' },
  { keyword: 'plumbing company near me', volume: 3200, position: 45, opportunity: 'high' },
];

// ── Autoplay Scenes ─────────────────────────────────────────────────────

interface Scene {
  id: string;
  title: string;
  subtitle: string;
  duration: number; // ms to show this scene
}

const SCENES: Scene[] = [
  { id: 'intro', title: 'Chatterbuzz SEO Automation Platform', subtitle: 'End-to-end SEO pipeline — fully automated', duration: 4000 },
  { id: 'onboard-1', title: 'Step 1: Client Onboarding', subtitle: 'Client submits business details & goals', duration: 4000 },
  { id: 'onboard-2', title: 'Step 1: Client Onboarding', subtitle: 'KPI targets & brand voice configured', duration: 3500 },
  { id: 'onboard-3', title: 'Step 1: Client Onboarding', subtitle: 'API credentials connected — ready to launch', duration: 3000 },
  { id: 'pipeline', title: 'Step 2: Data Pipeline', subtitle: '12-step automated pipeline runs', duration: 12000 },
  { id: 'wqa', title: 'Step 3: WQA Analysis', subtitle: 'Every page scored & prioritized', duration: 5000 },
  { id: 'keywords', title: 'Step 3: Keyword Opportunities', subtitle: 'Gap analysis reveals ranking opportunities', duration: 4000 },
  { id: 'content', title: 'Step 4: Content Engine', subtitle: 'AI generates quality-gated content', duration: 5000 },
  { id: 'dashboard', title: 'Step 5: Ops Dashboard', subtitle: 'Internal team monitors everything', duration: 5000 },
  { id: 'portal', title: 'Step 6: Client Portal', subtitle: 'Clients review & approve content', duration: 5000 },
  { id: 'finale', title: 'End-to-End Complete', subtitle: 'Onboarding to content delivery — fully automated', duration: 5000 },
];

// ── Helpers ──────────────────────────────────────────────────────────────

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

// ── Main Component ──────────────────────────────────────────────────────

export default function AutoplayDemo() {
  const [playing, setPlaying] = useState(false);
  const [sceneIndex, setSceneIndex] = useState(0);
  const [sceneProgress, setSceneProgress] = useState(0); // 0-100
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [wqaRevealCount, setWqaRevealCount] = useState(0);
  const [contentRevealCount, setContentRevealCount] = useState(0);
  const [contentApproved, setContentApproved] = useState<boolean[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const progressRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef(0);

  const scene = SCENES[sceneIndex];
  const totalScenes = SCENES.length;

  // Overall progress (0-100)
  const overallProgress = Math.round(((sceneIndex + sceneProgress / 100) / totalScenes) * 100);

  const clearTimers = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (progressRef.current) clearInterval(progressRef.current);
  }, []);

  const startScene = useCallback((index: number) => {
    if (index >= SCENES.length) {
      setPlaying(false);
      return;
    }

    setSceneIndex(index);
    setSceneProgress(0);
    startTimeRef.current = Date.now();

    const s = SCENES[index];

    // Reset scene-specific state
    if (s.id === 'pipeline') setPipelineStep(-1);
    if (s.id === 'wqa') setWqaRevealCount(0);
    if (s.id === 'content') {
      setContentRevealCount(0);
      setContentApproved([]);
    }

    // Progress bar
    progressRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      const pct = Math.min(100, (elapsed / s.duration) * 100);
      setSceneProgress(pct);
    }, 50);

    // Auto-advance
    timerRef.current = setTimeout(() => {
      if (progressRef.current) clearInterval(progressRef.current);
      setSceneProgress(100);
      startScene(index + 1);
    }, s.duration);
  }, []);

  // Pipeline animation
  useEffect(() => {
    if (!playing || scene?.id !== 'pipeline') return;
    const stepDuration = 900;
    const intervals: NodeJS.Timeout[] = [];
    PIPELINE_STEPS.forEach((_, i) => {
      intervals.push(setTimeout(() => setPipelineStep(i), i * stepDuration));
    });
    return () => intervals.forEach(clearTimeout);
  }, [playing, scene?.id]);

  // WQA reveal animation
  useEffect(() => {
    if (!playing || scene?.id !== 'wqa') return;
    const intervals: NodeJS.Timeout[] = [];
    WQA_RESULTS.forEach((_, i) => {
      intervals.push(setTimeout(() => setWqaRevealCount(i + 1), i * 500));
    });
    return () => intervals.forEach(clearTimeout);
  }, [playing, scene?.id]);

  // Content reveal + auto-approve animation
  useEffect(() => {
    if (!playing || scene?.id !== 'content') return;
    const intervals: NodeJS.Timeout[] = [];
    CONTENT_ITEMS.forEach((_, i) => {
      intervals.push(setTimeout(() => setContentRevealCount(i + 1), i * 600));
      intervals.push(setTimeout(() => {
        setContentApproved(prev => { const n = [...prev]; n[i] = true; return n; });
      }, i * 600 + 1200));
    });
    return () => intervals.forEach(clearTimeout);
  }, [playing, scene?.id]);

  function handlePlay() {
    clearTimers();
    setPlaying(true);
    startScene(0);
  }

  function handlePause() {
    clearTimers();
    setPlaying(false);
  }

  function handleRestart() {
    clearTimers();
    setSceneIndex(0);
    setSceneProgress(0);
    setPipelineStep(-1);
    setWqaRevealCount(0);
    setContentRevealCount(0);
    setContentApproved([]);
    setPlaying(true);
    startScene(0);
  }

  useEffect(() => {
    return () => clearTimers();
  }, [clearTimers]);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/demo" className="text-gray-400 hover:text-white text-sm">
              &larr; Manual Demo
            </Link>
            <span className="text-gray-700">|</span>
            <span className="text-lg font-bold tracking-tight">Chatterbuzz</span>
            <Badge color="purple">Autoplay</Badge>
          </div>
          <div className="flex items-center gap-3">
            {!playing ? (
              <button
                onClick={sceneIndex >= SCENES.length - 1 && sceneProgress >= 100 ? handleRestart : handlePlay}
                className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition text-sm"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.84A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.27l9.344-5.891a1.5 1.5 0 000-2.538L6.3 2.841z" />
                </svg>
                {sceneIndex >= SCENES.length - 1 && sceneProgress >= 100 ? 'Replay' : sceneIndex > 0 ? 'Resume' : 'Play'}
              </button>
            ) : (
              <button
                onClick={handlePause}
                className="flex items-center gap-2 px-5 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition text-sm"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M5.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75A.75.75 0 007.25 3h-1.5zm7 0a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75a.75.75 0 00-.75-.75h-1.5z" clipRule="evenodd" />
                </svg>
                Pause
              </button>
            )}
            <button
              onClick={handleRestart}
              className="px-3 py-2 text-gray-400 hover:text-white transition text-sm"
            >
              Restart
            </button>
          </div>
        </div>
        {/* Overall progress bar */}
        <div className="h-1 bg-gray-800">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-200"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Scene title bar */}
      <div className="bg-gray-900/50 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">{scene.title}</h2>
            <p className="text-sm text-gray-400 mt-0.5">{scene.subtitle}</p>
          </div>
          <div className="text-sm text-gray-500">
            {sceneIndex + 1} / {totalScenes}
          </div>
        </div>
        {/* Scene progress bar */}
        <div className="h-0.5 bg-gray-800">
          <div
            className="h-full bg-blue-500/50 transition-all duration-100"
            style={{ width: `${sceneProgress}%` }}
          />
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* ── INTRO ─────────────────────────────────────────── */}
        {scene.id === 'intro' && (
          <div className="text-center py-16 space-y-8 animate-fade-in">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Chatterbuzz SEO Automation
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Watch the complete platform flow — from client onboarding to automated content delivery.
            </p>
            <div className="grid grid-cols-3 md:grid-cols-6 gap-4 max-w-3xl mx-auto pt-4">
              {['Onboarding', 'Pipeline', 'WQA', 'Content', 'Dashboard', 'Portal'].map((step, i) => (
                <div key={step} className="text-center">
                  <div className="w-10 h-10 rounded-full bg-blue-600/20 border border-blue-800 flex items-center justify-center mx-auto text-sm font-bold text-blue-400">
                    {i + 1}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">{step}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── ONBOARDING 1: Business Details ──────────────── */}
        {scene.id === 'onboard-1' && (
          <div className="space-y-6 animate-fade-in">
            <div className="border border-gray-800 rounded-lg bg-gray-900 p-6">
              <div className="text-xs text-blue-400 font-medium mb-4 uppercase tracking-wider">Onboarding Wizard — Business Details</div>
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
            </div>
            <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4 text-sm text-blue-300">
              Data is validated with Zod and saved to the <code className="text-blue-200">clients</code> table in Supabase.
            </div>
          </div>
        )}

        {/* ── ONBOARDING 2: Goals & Brand Voice ──────────── */}
        {scene.id === 'onboard-2' && (
          <div className="space-y-6 animate-fade-in">
            <div className="border border-gray-800 rounded-lg bg-gray-900 p-6">
              <div className="text-xs text-blue-400 font-medium mb-4 uppercase tracking-wider">Onboarding Wizard — Goals & Brand Voice</div>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-800 border border-gray-700 rounded p-4 text-center">
                  <p className="text-3xl font-bold text-green-400">+{MOCK_CLIENT.goals.kpi_targets.organic_sessions_growth}%</p>
                  <p className="text-xs text-gray-400 mt-1">Organic Traffic Target</p>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded p-4 text-center">
                  <p className="text-3xl font-bold text-blue-400">{MOCK_CLIENT.goals.kpi_targets.keyword_rankings_top10}</p>
                  <p className="text-xs text-gray-400 mt-1">Keywords in Top 10</p>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded p-4 text-center">
                  <p className="text-3xl font-bold text-purple-400">+{MOCK_CLIENT.goals.kpi_targets.conversions_growth}%</p>
                  <p className="text-xs text-gray-400 mt-1">Conversions Growth</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Tone</label>
                  <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.brand_voice.tone}</div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Style</label>
                  <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">{MOCK_CLIENT.brand_voice.style}</div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Topics to Avoid</label>
                  <div className="flex gap-2">
                    {MOCK_CLIENT.brand_voice.avoid.map((t) => (
                      <Badge key={t} color="red">{t}</Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── ONBOARDING 3: API Connected ─────────────────── */}
        {scene.id === 'onboard-3' && (
          <div className="space-y-6 animate-fade-in">
            <div className="border border-gray-800 rounded-lg bg-gray-900 p-6">
              <div className="text-xs text-blue-400 font-medium mb-4 uppercase tracking-wider">Onboarding Wizard — Integrations Connected</div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { name: 'Google Search Console', status: 'Connected' },
                  { name: 'Google Analytics 4', status: 'Connected' },
                  { name: 'WordPress CMS', status: 'Connected' },
                  { name: 'Google Business Profile', status: 'Connected' },
                ].map((api) => (
                  <div key={api.name} className="bg-gray-800 border border-gray-700 rounded px-4 py-3 flex items-center justify-between">
                    <span className="text-sm">{api.name}</span>
                    <span className="flex items-center gap-2 text-sm text-green-400">
                      <span className="w-2 h-2 rounded-full bg-green-400" /> {api.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="border border-green-800 bg-green-900/20 rounded-lg p-4 text-center">
              <p className="text-green-300 font-medium">Client onboarded successfully — triggering first pipeline run...</p>
            </div>
          </div>
        )}

        {/* ── PIPELINE ────────────────────────────────────── */}
        {scene.id === 'pipeline' && (
          <div className="space-y-4 animate-fade-in">
            <div className="border border-gray-800 rounded-lg bg-gray-900 p-6">
              <div className="text-xs text-purple-400 font-medium mb-4 uppercase tracking-wider">Pipeline: {MOCK_CLIENT.name}</div>
              <div className="space-y-2">
                {PIPELINE_STEPS.map((step, i) => {
                  const isDone = i < pipelineStep;
                  const isActive = i === pipelineStep;
                  const isPending = i > pipelineStep;
                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-3 px-3 py-2 rounded transition-all duration-300 ${
                        isActive ? 'bg-blue-900/30 border border-blue-800' :
                        isDone ? 'bg-gray-800/50' : ''
                      }`}
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-all ${
                        isDone ? 'bg-green-600 text-white' :
                        isActive ? 'bg-blue-600 text-white animate-pulse' :
                        'bg-gray-800 text-gray-500'
                      }`}>
                        {isDone ? '\u2713' : i + 1}
                      </div>
                      <span className={`text-sm transition-colors ${isPending ? 'text-gray-600' : isDone ? 'text-gray-400' : 'text-white font-medium'}`}>
                        {step}
                      </span>
                      {isActive && <span className="ml-auto text-xs text-blue-400 animate-pulse">Processing...</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── WQA ANALYSIS ────────────────────────────────── */}
        {scene.id === 'wqa' && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-4 gap-4">
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Pages Analyzed</p>
                <p className="text-3xl font-bold mt-1">{wqaRevealCount}</p>
              </div>
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">High Priority</p>
                <p className="text-3xl font-bold mt-1 text-red-400">
                  {WQA_RESULTS.slice(0, wqaRevealCount).filter(r => r.priority === 'high').length}
                </p>
              </div>
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Avg Score</p>
                <p className="text-3xl font-bold mt-1 text-yellow-400">
                  {wqaRevealCount > 0 ? Math.round(WQA_RESULTS.slice(0, wqaRevealCount).reduce((s, r) => s + r.score, 0) / wqaRevealCount) : 0}
                </p>
              </div>
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Rewrites Needed</p>
                <p className="text-3xl font-bold mt-1 text-blue-400">
                  {WQA_RESULTS.slice(0, wqaRevealCount).filter(r => r.action === 'Rewrite').length}
                </p>
              </div>
            </div>
            <div className="border border-gray-800 rounded-lg bg-gray-900 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">URL</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Score</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Priority</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Action</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Issues</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {WQA_RESULTS.slice(0, wqaRevealCount).map((r) => (
                    <tr key={r.url} className="animate-slide-in">
                      <td className="px-4 py-2 text-blue-400">{r.url}</td>
                      <td className="px-4 py-2">
                        <span className={`font-bold ${r.score >= 70 ? 'text-green-400' : r.score >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {r.score}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        <Badge color={r.priority === 'high' ? 'red' : r.priority === 'medium' ? 'yellow' : 'green'}>{r.priority}</Badge>
                      </td>
                      <td className="px-4 py-2">{r.action}</td>
                      <td className="px-4 py-2">
                        <div className="flex flex-wrap gap-1">
                          {r.issues.map((issue) => (
                            <span key={issue} className="text-xs bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">{issue}</span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── KEYWORDS ────────────────────────────────────── */}
        {scene.id === 'keywords' && (
          <div className="space-y-4 animate-fade-in">
            <div className="border border-gray-800 rounded-lg bg-gray-900 overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-800 font-semibold text-sm">Keyword Opportunities</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Keyword</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Monthly Volume</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Current Position</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Opportunity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {KEYWORDS.map((kw) => (
                    <tr key={kw.keyword}>
                      <td className="px-4 py-2 font-medium">{kw.keyword}</td>
                      <td className="px-4 py-2 text-gray-300">{kw.volume.toLocaleString()}</td>
                      <td className="px-4 py-2 text-gray-400">{kw.position ?? 'Not ranking'}</td>
                      <td className="px-4 py-2">
                        <Badge color={kw.opportunity === 'high' ? 'red' : kw.opportunity === 'new' ? 'purple' : 'yellow'}>
                          {kw.opportunity}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── CONTENT ENGINE ──────────────────────────────── */}
        {scene.id === 'content' && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-3 gap-4">
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Generated</p>
                <p className="text-3xl font-bold mt-1">{contentRevealCount}</p>
              </div>
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Quality Gate Passed</p>
                <p className="text-3xl font-bold mt-1 text-green-400">{contentApproved.filter(Boolean).length}</p>
              </div>
              <div className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                <p className="text-sm text-gray-400">Avg Quality Score</p>
                <p className="text-3xl font-bold mt-1 text-blue-400">
                  {contentRevealCount > 0 ? Math.round(CONTENT_ITEMS.slice(0, contentRevealCount).reduce((s, c) => s + c.score, 0) / contentRevealCount) : 0}
                </p>
              </div>
            </div>
            <div className="border border-gray-800 rounded-lg bg-gray-900 p-5">
              <div className="text-xs text-blue-400 font-medium mb-4 uppercase tracking-wider">Content Queue — Quality Gate (70/100)</div>
              <div className="space-y-3">
                {CONTENT_ITEMS.slice(0, contentRevealCount).map((item, i) => (
                  <div key={i} className="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-3 animate-slide-in">
                    <div className="flex-1">
                      <p className="font-medium text-sm">{item.title}</p>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                        <span>{item.type}</span>
                        <span>{item.words} words</span>
                        <span>Quality: <span className={item.score >= 80 ? 'text-green-400' : 'text-yellow-400'}>{item.score}/100</span></span>
                      </div>
                    </div>
                    <Badge color={contentApproved[i] ? 'green' : 'yellow'}>
                      {contentApproved[i] ? 'Passed' : 'Scoring...'}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── OPS DASHBOARD ───────────────────────────────── */}
        {scene.id === 'dashboard' && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: 'Active Clients', value: '12', color: 'text-white' },
                { label: 'Pipeline Runs (24h)', value: '34', color: 'text-blue-400' },
                { label: 'Content Pending Review', value: '8', color: 'text-yellow-400' },
                { label: 'Active Alerts', value: '2', color: 'text-red-400' },
              ].map((card) => (
                <div key={card.label} className="border border-gray-800 rounded-lg bg-gray-900 p-4">
                  <p className="text-sm text-gray-400">{card.label}</p>
                  <p className={`text-3xl font-bold mt-1 ${card.color}`}>{card.value}</p>
                </div>
              ))}
            </div>
            <div className="border border-gray-800 rounded-lg bg-gray-900 overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-800 font-semibold text-sm">All Clients</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Client</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Domain</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Status</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Last Pipeline</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Content Queue</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {[
                    { name: 'Acme Plumbing Co.', domain: 'acmeplumbing.com', status: 'active', lastRun: '2h ago', queue: 6 },
                    { name: 'Sunset Dental', domain: 'sunsetdental.com', status: 'active', lastRun: '5h ago', queue: 3 },
                    { name: 'Peak Fitness ATL', domain: 'peakfitnessatl.com', status: 'active', lastRun: '1d ago', queue: 0 },
                    { name: 'GreenLeaf Landscaping', domain: 'greenleaflandscape.com', status: 'onboarding', lastRun: '-', queue: 0 },
                  ].map((c) => (
                    <tr key={c.name}>
                      <td className="px-4 py-2 text-blue-400 font-medium">{c.name}</td>
                      <td className="px-4 py-2 text-gray-300">{c.domain}</td>
                      <td className="px-4 py-2"><Badge color={c.status === 'active' ? 'green' : 'yellow'}>{c.status}</Badge></td>
                      <td className="px-4 py-2 text-gray-400">{c.lastRun}</td>
                      <td className="px-4 py-2">{c.queue > 0 ? <Badge color="yellow">{c.queue} pending</Badge> : <span className="text-gray-600">-</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="border border-red-900/50 bg-red-900/10 rounded-lg p-4">
                <p className="text-sm font-medium text-red-300">Alert: Pipeline Failed — Acme Plumbing</p>
                <p className="text-xs text-gray-400 mt-1">Index monitor timed out. Auto-retry in 1h.</p>
              </div>
              <div className="border border-yellow-900/50 bg-yellow-900/10 rounded-lg p-4">
                <p className="text-sm font-medium text-yellow-300">Alert: Content Below Threshold</p>
                <p className="text-xs text-gray-400 mt-1">1 piece for Sunset Dental scored 62/100. Queued for regeneration.</p>
              </div>
            </div>
          </div>
        )}

        {/* ── CLIENT PORTAL ───────────────────────────────── */}
        {scene.id === 'portal' && (
          <div className="animate-fade-in">
            <div className="bg-white text-gray-900 rounded-xl overflow-hidden border shadow-lg">
              <div className="bg-gray-50 border-b px-6 py-4 flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-lg">{MOCK_CLIENT.name}</h3>
                  <p className="text-sm text-gray-500">{MOCK_CLIENT.domain}</p>
                </div>
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700 border border-green-200">Active</span>
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
                    <p className="text-3xl font-bold text-purple-600">5</p>
                    <p className="text-sm text-gray-500 mt-1">Content Pieces Ready</p>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Content Awaiting Your Review</h4>
                  <div className="space-y-2">
                    {CONTENT_ITEMS.slice(0, 3).map((item, i) => (
                      <div key={i} className="border rounded-lg p-3 flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{item.title}</p>
                          <p className="text-xs text-gray-500">{item.type} &middot; {item.words} words &middot; Quality: {item.score}/100</p>
                        </div>
                        <div className="flex gap-2">
                          <span className="px-3 py-1.5 bg-green-600 text-white rounded text-sm">Approve</span>
                          <span className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded text-sm">Reject</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-3">Monthly Report — February 2026</h4>
                  <div className="border rounded-lg p-4 bg-gray-50 grid grid-cols-4 gap-4 text-sm">
                    <div><p className="text-gray-500">Pages Optimized</p><p className="font-bold text-lg">5</p></div>
                    <div><p className="text-gray-500">New Content Published</p><p className="font-bold text-lg">3</p></div>
                    <div><p className="text-gray-500">Avg. WQA Score</p><p className="font-bold text-lg">72 (+18)</p></div>
                    <div><p className="text-gray-500">Impressions</p><p className="font-bold text-lg">24.3K</p></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── FINALE ──────────────────────────────────────── */}
        {scene.id === 'finale' && (
          <div className="text-center py-16 animate-fade-in">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-12 max-w-2xl mx-auto">
              <h2 className="text-3xl font-bold mb-3">End-to-End Demo Complete</h2>
              <p className="text-blue-100 text-lg mb-8">
                From client onboarding to automated content delivery — fully automated, quality-gated, and client-facing.
              </p>
              <div className="grid grid-cols-3 gap-4 mb-8 text-center">
                <div className="bg-white/10 rounded-lg p-4">
                  <p className="text-2xl font-bold">18</p>
                  <p className="text-sm text-blue-200">Supabase Tables</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4">
                  <p className="text-2xl font-bold">12</p>
                  <p className="text-sm text-blue-200">Pipeline Steps</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4">
                  <p className="text-2xl font-bold">10</p>
                  <p className="text-sm text-blue-200">n8n Workflows</p>
                </div>
              </div>
              <div className="flex justify-center gap-4">
                <button
                  onClick={handleRestart}
                  className="px-6 py-2 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-100 transition"
                >
                  Watch Again
                </button>
                <Link
                  href="/demo"
                  className="px-6 py-2 bg-white/20 text-white rounded-lg font-medium hover:bg-white/30 transition"
                >
                  Interactive Demo
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out;
        }
        .animate-slide-in {
          animation: slideIn 0.3s ease-out;
        }
      `}</style>
    </main>
  );
}
