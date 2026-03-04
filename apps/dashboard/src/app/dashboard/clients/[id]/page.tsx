export default function ClientDetailPage({ params }: { params: { id: string } }) {
  const tabs = ['Pipeline', 'WQA Results', 'Content Queue', 'Keywords', 'Schema', 'Reports', 'Settings'];

  return (
    <main className="max-w-7xl mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">Client Details</h1>
      <p className="text-gray-400 mb-6">Client ID: {params.id}</p>

      <div className="flex gap-2 mb-8 border-b border-gray-800 pb-2">
        {tabs.map((tab) => (
          <button
            key={tab}
            className="px-4 py-2 text-sm rounded-t hover:bg-gray-800 transition"
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="border border-gray-800 rounded-lg p-8 bg-gray-900">
        <p className="text-gray-500">Select a tab to view details</p>
      </div>
    </main>
  );
}
