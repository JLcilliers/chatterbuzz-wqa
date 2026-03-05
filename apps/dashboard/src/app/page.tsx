'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@/lib/supabase';
import Link from 'next/link';

interface Client {
  id: string;
  name: string;
  domain: string;
  status: string;
  created_at: string;
}

export default function DashboardHome() {
  const [stats, setStats] = useState({
    activeClients: 0,
    pipelineRuns24h: 0,
    contentPendingReview: 0,
    activeAlerts: 0,
  });
  const [clients, setClients] = useState<Client[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      const supabase = createClient();

      const [clientsRes, pipelineRes, contentRes, alertsRes] = await Promise.all([
        supabase.from('clients').select('id, name, domain, status, created_at'),
        supabase
          .from('pipeline_runs')
          .select('id', { count: 'exact', head: true })
          .gte('started_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()),
        supabase
          .from('content_queue')
          .select('id', { count: 'exact', head: true })
          .eq('status', 'review'),
        supabase
          .from('pipeline_runs')
          .select('id', { count: 'exact', head: true })
          .eq('status', 'failed'),
      ]);

      const allClients = clientsRes.data ?? [];
      const activeClients = allClients.filter((c) => c.status === 'active');

      setStats({
        activeClients: activeClients.length,
        pipelineRuns24h: pipelineRes.count ?? 0,
        contentPendingReview: contentRes.count ?? 0,
        activeAlerts: alertsRes.count ?? 0,
      });
      setClients(allClients);
      setLoading(false);
    }

    fetchData();
  }, []);

  const cards = [
    { label: 'Active Clients', value: stats.activeClients },
    { label: 'Pipeline Runs (24h)', value: stats.pipelineRuns24h },
    { label: 'Content Pending Review', value: stats.contentPendingReview },
    { label: 'Active Alerts', value: stats.activeAlerts },
  ];

  return (
    <main className="max-w-7xl mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Ops Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        {cards.map((card) => (
          <div key={card.label} className="border border-gray-800 rounded-lg p-6 bg-gray-900">
            <p className="text-sm text-gray-400">{card.label}</p>
            <p className="text-3xl font-bold mt-2">
              {loading ? '...' : card.value}
            </p>
          </div>
        ))}
      </div>

      <h2 className="text-xl font-semibold mb-4">Clients</h2>
      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : clients.length === 0 ? (
        <p className="text-gray-500">No clients yet. Complete onboarding to add your first client.</p>
      ) : (
        <div className="border border-gray-800 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900">
              <tr>
                <th className="text-left px-4 py-3 text-gray-400 font-medium">Name</th>
                <th className="text-left px-4 py-3 text-gray-400 font-medium">Domain</th>
                <th className="text-left px-4 py-3 text-gray-400 font-medium">Status</th>
                <th className="text-left px-4 py-3 text-gray-400 font-medium">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {clients.map((client) => (
                <tr key={client.id} className="hover:bg-gray-900/50">
                  <td className="px-4 py-3">
                    <Link
                      href={`/dashboard/clients/${client.id}`}
                      className="text-blue-400 hover:underline"
                    >
                      {client.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-300">{client.domain}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                        client.status === 'active'
                          ? 'bg-green-900/50 text-green-400'
                          : client.status === 'onboarding'
                          ? 'bg-yellow-900/50 text-yellow-400'
                          : 'bg-gray-800 text-gray-400'
                      }`}
                    >
                      {client.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-400">
                    {new Date(client.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
