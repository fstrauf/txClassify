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
          DEFAULT: "#4361ee", // Bright blue for primary actions
          light: "#3B82F6",
          dark: "#1D4ED8",
        },
        secondary: {
          DEFAULT: "#b5179e", // Magenta/pink for secondary elements
          light: "#d31eb3", // Lighter magenta
          dark: "#8a1277", // Darker magenta
        },
        background: {
          DEFAULT: "#F9FAFB",
          dark: "#1F2937",
        },
        surface: {
          DEFAULT: "#FFFFFF",
          dark: "#374151",
        },
        accent: {
          DEFAULT: "#f72585", // Purple for accents
          light: "#A78BFA",
          dark: "#7C3AED",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        soft: "0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)",
        glow: "0 0 20px rgba(37, 99, 235, 0.15)",
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
      },
      animation: {
        gradient: "gradient 8s linear infinite",
        fadeIn: "fadeIn 0.5s ease-in-out",
        pulse: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        gradient: {
          "0%, 100%": {
            "background-size": "200% 200%",
            "background-position": "left center",
          },
          "50%": {
            "background-size": "200% 200%",
            "background-position": "right center",
          },
        },
        fadeIn: {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        pulse: {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.8 },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
  // variants: {
  //   extend: {
  //     opacity: ["disabled"],
  //   },
  // },
};
