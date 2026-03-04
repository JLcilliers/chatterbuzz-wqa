export default function DashboardHome() {
  return (
    <main className="max-w-7xl mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Ops Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Active Clients', value: '—' },
          { label: 'Pipeline Runs (24h)', value: '—' },
          { label: 'Content Pending Review', value: '—' },
          { label: 'Active Alerts', value: '—' },
        ].map((card) => (
          <div key={card.label} className="border border-gray-800 rounded-lg p-6 bg-gray-900">
            <p className="text-sm text-gray-400">{card.label}</p>
            <p className="text-3xl font-bold mt-2">{card.value}</p>
          </div>
        ))}
      </div>
    </main>
  );
}
