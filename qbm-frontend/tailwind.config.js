/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // QBM Brand Colors
        qbm: {
          primary: '#065f46',    // Emerald 800
          secondary: '#059669',  // Emerald 600
          accent: '#10b981',     // Emerald 500
          light: '#d1fae5',      // Emerald 100
          dark: '#022c22',       // Emerald 950
        },
      },
      fontFamily: {
        // Arabic-friendly fonts
        arabic: ['Amiri', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
