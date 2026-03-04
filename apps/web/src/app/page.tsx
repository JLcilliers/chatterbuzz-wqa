import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-4">Chatterbuzz SEO Platform</h1>
      <p className="text-gray-600 mb-8">Client onboarding and content management portal</p>
      <div className="flex gap-4">
        <Link
          href="/onboard"
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Start Onboarding
        </Link>
        <Link
          href="/portal"
          className="px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
        >
          Client Portal
        </Link>
      </div>
    </main>
  );
}
