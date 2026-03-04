import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Chatterbuzz SEO Platform',
  description: 'Client onboarding and portal',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 antialiased">{children}</body>
    </html>
  );
}
