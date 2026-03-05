'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@/lib/supabase';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

type Tab = 'Dashboard' | 'Content Review' | 'Reports' | 'Settings';

export default function PortalPage({ params }: { params: { clientId: string } }) {
  const { clientId } = params;
  const [activeTab, setActiveTab] = useState<Tab>('Dashboard');
  const [client, setClient] = useState<any>(null);
  const [pipelineRuns, setPipelineRuns] = useState<any[]>([]);
  const [contentQueue, setContentQueue] = useState<any[]>([]);
  const [monthlyReports, setMonthlyReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      const supabase = createClient();

      const [clientRes, pipelineRes, contentRes, reportsRes] = await Promise.all([
        supabase.from('clients').select('*').eq('id', clientId).single(),
        supabase
          .from('pipeline_runs')
          .select('*')
          .eq('client_id', clientId)
          .order('started_at', { ascending: false })
          .limit(5),
        supabase
          .from('content_queue')
          .select('*')
          .eq('client_id', clientId)
          .order('created_at', { ascending: false })
          .limit(50),
        supabase
          .from('monthly_reports')
          .select('*')
          .eq('client_id', clientId)
          .order('report_month', { ascending: false })
          .limit(12),
      ]);

      setClient(clientRes.data);
      setPipelineRuns(pipelineRes.data ?? []);
      setContentQueue(contentRes.data ?? []);
      setMonthlyReports(reportsRes.data ?? []);
      setLoading(false);
    }

    fetchData();
  }, [clientId]);

  async function updateContentStatus(contentId: string, status: 'approved' | 'rejected') {
    const supabase = createClient();
    await supabase.from('content_queue').update({ status }).eq('id', contentId);
    setContentQueue((prev) =>
      prev.map((item) => (item.id === contentId ? { ...item, status } : item))
    );
  }

  const tabs: Tab[] = ['Dashboard', 'Content Review', 'Reports', 'Settings'];

  if (loading) {
    return (
      <main className="max-w-6xl mx-auto py-8 px-4">
        <p className="text-gray-500">Loading portal...</p>
      </main>
    );
  }

  if (!client) {
    return (
      <main className="max-w-6xl mx-auto py-8 px-4">
        <p className="text-red-600">Client not found.</p>
      </main>
    );
  }

  return (
    <main className="max-w-6xl mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-2">{client.name}</h1>
      <p className="text-gray-500 mb-6">{client.domain}</p>

      <div className="flex gap-2 mb-8 border-b pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm rounded-t transition ${
              activeTab === tab
                ? 'bg-gray-100 text-gray-900 font-medium'
                : 'text-gray-500 hover:bg-gray-50'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'Dashboard' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border rounded-lg p-5 bg-white">
              <p className="text-sm text-gray-500">Status</p>
              <p className="text-lg font-semibold mt-1 capitalize">{client.status}</p>
            </div>
            <div className="border rounded-lg p-5 bg-white">
              <p className="text-sm text-gray-500">Industry</p>
              <p className="text-lg font-semibold mt-1">{client.industry}</p>
            </div>
            <div className="border rounded-lg p-5 bg-white">
              <p className="text-sm text-gray-500">Content in Queue</p>
              <p className="text-lg font-semibold mt-1">{contentQueue.length}</p>
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-3">Recent Pipeline Activity</h3>
            {pipelineRuns.length === 0 ? (
              <p className="text-gray-500 text-sm">No pipeline runs yet.</p>
            ) : (
              <div className="border rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="text-left px-4 py-2 text-gray-500 font-medium">Type</th>
                      <th className="text-left px-4 py-2 text-gray-500 font-medium">Status</th>
                      <th className="text-left px-4 py-2 text-gray-500 font-medium">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {pipelineRuns.map((run) => (
                      <tr key={run.id}>
                        <td className="px-4 py-2">{run.pipeline_type}</td>
                        <td className="px-4 py-2">
                          <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                            run.status === 'completed' || run.status === 'success'
                              ? 'bg-green-100 text-green-700'
                              : run.status === 'failed'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-yellow-100 text-yellow-700'
                          }`}>
                            {run.status}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-gray-500">
                          {run.started_at ? new Date(run.started_at).toLocaleDateString() : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'Content Review' && (
        <div>
          {contentQueue.length === 0 ? (
            <p className="text-gray-500">No content items to review.</p>
          ) : (
            <div className="space-y-4">
              {contentQueue.map((item) => (
                <div key={item.id} className="border rounded-lg p-4 bg-white">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium">{item.title || item.target_url || 'Untitled'}</h4>
                      <p className="text-sm text-gray-500 mt-1">
                        {item.content_type || 'Content'} &middot;{' '}
                        <span className={`font-medium ${
                          item.status === 'approved' ? 'text-green-600' :
                          item.status === 'rejected' ? 'text-red-600' :
                          item.status === 'review' ? 'text-yellow-600' :
                          'text-gray-500'
                        }`}>
                          {item.status}
                        </span>
                      </p>
                    </div>
                    {item.status === 'review' && (
                      <div className="flex gap-2">
                        <button
                          onClick={() => updateContentStatus(item.id, 'approved')}
                          className="px-3 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => updateContentStatus(item.id, 'rejected')}
                          className="px-3 py-1.5 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                        >
                          Reject
                        </button>
                      </div>
                    )}
                  </div>
                  {item.content_body && (
                    <p className="text-sm text-gray-600 mt-3 line-clamp-3">{item.content_body}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'Reports' && (
        <div>
          {monthlyReports.length === 0 ? (
            <p className="text-gray-500">No reports available yet. Reports are generated monthly after your first pipeline run.</p>
          ) : (
            <div className="space-y-4">
              {monthlyReports.map((report) => (
                <div key={report.id} className="border rounded-lg p-5 bg-white">
                  <h4 className="font-medium">{report.report_month}</h4>
                  {report.summary && (
                    <p className="text-sm text-gray-600 mt-2">{report.summary}</p>
                  )}
                  {report.metrics && (
                    <pre className="mt-3 bg-gray-50 rounded p-3 text-xs overflow-auto">
                      {JSON.stringify(report.metrics, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'Settings' && (
        <div className="space-y-6">
          <div className="border rounded-lg p-5 bg-white">
            <h3 className="font-semibold mb-4">Client Information</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Business Name</p>
                <p className="font-medium">{client.name}</p>
              </div>
              <div>
                <p className="text-gray-500">Domain</p>
                <p className="font-medium">{client.domain}</p>
              </div>
              <div>
                <p className="text-gray-500">Industry</p>
                <p className="font-medium">{client.industry}</p>
              </div>
              <div>
                <p className="text-gray-500">Business Type</p>
                <p className="font-medium">{client.business_type}</p>
              </div>
              <div>
                <p className="text-gray-500">Target Locations</p>
                <p className="font-medium">{client.target_locations?.join(', ') || '—'}</p>
              </div>
              <div>
                <p className="text-gray-500">Member Since</p>
                <p className="font-medium">{new Date(client.created_at).toLocaleDateString()}</p>
              </div>
            </div>
          </div>
          {client.brand_voice && (
            <div className="border rounded-lg p-5 bg-white">
              <h3 className="font-semibold mb-3">Brand Voice</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Tone</p>
                  <p className="font-medium">{client.brand_voice.tone}</p>
                </div>
                <div>
                  <p className="text-gray-500">Style</p>
                  <p className="font-medium">{client.brand_voice.style}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
