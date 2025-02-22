/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Modern color palette inspired by the reference sites
        primary: {
          DEFAULT: '#2563EB', // Bright blue for primary actions
          light: '#3B82F6',
          dark: '#1D4ED8',
        },
        secondary: {
          DEFAULT: '#10B981', // Success green
          light: '#34D399',
          dark: '#059669',
        },
        background: {
          DEFAULT: '#F9FAFB',
          dark: '#1F2937',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          dark: '#374151',
        },
        accent: {
          DEFAULT: '#8B5CF6', // Purple for accents
          light: '#A78BFA',
          dark: '#7C3AED',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'glow': '0 0 20px rgba(37, 99, 235, 0.15)',
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
      animation: {
        'gradient': 'gradient 8s linear infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center',
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center',
          },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/typography"),
  ],
  // variants: {
  //   extend: {
  //     opacity: ["disabled"],
  //   },
  // },
};
