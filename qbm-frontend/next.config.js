/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React Strict Mode for better development experience
  reactStrictMode: true,
  
  // Environment variables that should be available on the client
  env: {
    NEXT_PUBLIC_APP_NAME: 'QBM Research Assistant',
  },
  
  // Headers for Arabic text support
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Content-Language',
            value: 'ar, en',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
