'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@/lib/supabase';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface ClientInfo {
  id: string;
  name: string;
  domain: string;
  status: string;
  industry: string;
  goals: Record<string, unknown>;
  brand_voice: Record<string, unknown>;
  created_at: string;
}

export default function ClientDetailPage({ params }: { params: { id: string } }) {
  const clientId = params.id;
  const tabs = ['Pipeline', 'WQA Results', 'Content Queue', 'Settings'] as const;
  type Tab = typeof tabs[number];

  const [activeTab, setActiveTab] = useState<Tab>('Pipeline');
  const [client, setClient] = useState<ClientInfo | null>(null);
  const [pipelineRuns, setPipelineRuns] = useState<any[]>([]);
  const [wqaResults, setWqaResults] = useState<any[]>([]);
  const [contentQueue, setContentQueue] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState('');

  useEffect(() => {
    async function fetchData() {
      const supabase = createClient();

      const [clientRes, pipelineRes, wqaRes, contentRes] = await Promise.all([
        supabase.from('clients').select('*').eq('id', clientId).single(),
        supabase
          .from('pipeline_runs')
          .select('*')
          .eq('client_id', clientId)
          .order('started_at', { ascending: false })
          .limit(20),
        supabase
          .from('wqa_results')
          .select('*')
          .eq('client_id', clientId)
          .order('created_at', { ascending: false })
          .limit(50),
        supabase
          .from('content_queue')
          .select('*')
          .eq('client_id', clientId)
          .order('created_at', { ascending: false })
          .limit(50),
      ]);

      setClient(clientRes.data);
      setPipelineRuns(pipelineRes.data ?? []);
      setWqaResults(wqaRes.data ?? []);
      setContentQueue(contentRes.data ?? []);
      setLoading(false);
    }

    fetchData();
  }, [clientId]);

  async function triggerAction(endpoint: string, label: string) {
    if (!API_URL) {
      alert('API_URL not configured. Set NEXT_PUBLIC_API_URL environment variable.');
      return;
    }
    setActionLoading(label);
    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': prompt('Enter API key:') || '',
        },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        alert(`Error: ${err.detail || res.statusText}`);
      } else {
        alert(`${label} triggered successfully`);
        window.location.reload();
      }
    } catch (err) {
      alert(`Failed to trigger ${label}: ${err}`);
    } finally {
      setActionLoading('');
    }
  }

  const statusColor = (status: string) => {
    const colors: Record<string, string> = {
      completed: 'bg-green-900/50 text-green-400',
      success: 'bg-green-900/50 text-green-400',
      running: 'bg-blue-900/50 text-blue-400',
      pending: 'bg-yellow-900/50 text-yellow-400',
      review: 'bg-yellow-900/50 text-yellow-400',
      failed: 'bg-red-900/50 text-red-400',
      published: 'bg-green-900/50 text-green-400',
      draft: 'bg-gray-800 text-gray-400',
    };
    return colors[status] || 'bg-gray-800 text-gray-400';
  };

  if (loading) {
    return (
      <main className="max-w-7xl mx-auto py-8 px-4">
        <p className="text-gray-500">Loading client data...</p>
      </main>
    );
  }

  return (
    <main className="max-w-7xl mx-auto py-8 px-4 bg-gray-950 text-gray-100 min-h-screen">
      <Link href="/dashboard" className="text-blue-400 hover:underline text-sm mb-4 inline-block">
        &larr; Back to Dashboard
      </Link>

      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">{client?.name || 'Unknown Client'}</h1>
          <p className="text-gray-400">{client?.domain} &middot; {client?.industry}</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => triggerAction(`/wqa/${clientId}/run`, 'Run WQA')}
            disabled={!!actionLoading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium disabled:opacity-50"
          >
            {actionLoading === 'Run WQA' ? 'Running...' : 'Run WQA'}
          </button>
          <button
            onClick={() => triggerAction(`/pipeline/${clientId}/run`, 'Trigger Pipeline')}
            disabled={!!actionLoading}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium disabled:opacity-50"
          >
            {actionLoading === 'Trigger Pipeline' ? 'Running...' : 'Trigger Pipeline'}
          </button>
        </div>
      </div>

      <div className="flex gap-2 mb-8 border-b border-gray-800 pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm rounded-t transition ${
              activeTab === tab
                ? 'bg-gray-800 text-white font-medium'
                : 'hover:bg-gray-800/50 text-gray-400'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="border border-gray-800 rounded-lg bg-gray-900 overflow-hidden">
        {activeTab === 'Pipeline' && (
          <div>
            {pipelineRuns.length === 0 ? (
              <p className="p-6 text-gray-500">No pipeline runs yet.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Type</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Status</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Started</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Finished</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {pipelineRuns.map((run) => (
                    <tr key={run.id}>
                      <td className="px-4 py-3">{run.pipeline_type}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${statusColor(run.status)}`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-400">
                        {run.started_at ? new Date(run.started_at).toLocaleString() : '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-400">
                        {run.finished_at ? new Date(run.finished_at).toLocaleString() : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'WQA Results' && (
          <div>
            {wqaResults.length === 0 ? (
              <p className="p-6 text-gray-500">No WQA results yet. Run a WQA analysis to get started.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">URL</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Score</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Priority</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {wqaResults.map((result) => (
                    <tr key={result.id}>
                      <td className="px-4 py-3 text-blue-400 max-w-xs truncate">{result.url}</td>
                      <td className="px-4 py-3">{result.overall_score ?? '—'}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                          result.priority === 'high' ? 'bg-red-900/50 text-red-400' :
                          result.priority === 'medium' ? 'bg-yellow-900/50 text-yellow-400' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                          {result.priority || '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-400">
                        {new Date(result.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'Content Queue' && (
          <div>
            {contentQueue.length === 0 ? (
              <p className="p-6 text-gray-500">No content in queue.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Title</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Type</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Status</th>
                    <th className="text-left px-4 py-3 text-gray-400 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {contentQueue.map((item) => (
                    <tr key={item.id}>
                      <td className="px-4 py-3 max-w-xs truncate">{item.title || item.target_url || '—'}</td>
                      <td className="px-4 py-3 text-gray-300">{item.content_type || '—'}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${statusColor(item.status)}`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-400">
                        {new Date(item.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'Settings' && (
          <div className="p-6 space-y-4">
            <h3 className="font-semibold text-lg">Client Settings</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-400">Status</p>
                <p>{client?.status}</p>
              </div>
              <div>
                <p className="text-gray-400">Industry</p>
                <p>{client?.industry}</p>
              </div>
              <div>
                <p className="text-gray-400">Domain</p>
                <p>{client?.domain}</p>
              </div>
              <div>
                <p className="text-gray-400">Created</p>
                <p>{client?.created_at ? new Date(client.created_at).toLocaleDateString() : '—'}</p>
              </div>
            </div>
            {client?.goals && (
              <div>
                <p className="text-gray-400 text-sm mb-1">Goals</p>
                <pre className="bg-gray-950 rounded p-3 text-xs overflow-auto">
                  {JSON.stringify(client.goals, null, 2)}
                </pre>
              </div>
            )}
            {client?.brand_voice && (
              <div>
                <p className="text-gray-400 text-sm mb-1">Brand Voice</p>
                <pre className="bg-gray-950 rounded p-3 text-xs overflow-auto">
                  {JSON.stringify(client.brand_voice, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
