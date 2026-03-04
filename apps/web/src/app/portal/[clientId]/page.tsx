export default function PortalPage({ params }: { params: { clientId: string } }) {
  return (
    <main className="max-w-6xl mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">Client Portal</h1>
      <p className="text-gray-600 mb-8">Client ID: {params.clientId}</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {['Dashboard', 'Content Review', 'Reports', 'Settings'].map((tab) => (
          <div key={tab} className="border rounded-lg p-6 bg-white hover:shadow-md transition">
            <h3 className="font-semibold">{tab}</h3>
            <p className="text-sm text-gray-500 mt-2">Coming soon</p>
          </div>
        ))}
      </div>
    </main>
  );
}
