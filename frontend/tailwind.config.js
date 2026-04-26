/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        providencia: {
          blue: '#004987',
          green: '#61A60E',
          dark: '#121212',
          darker: '#0a0a0a',
          light: '#e0e0e0',
          gray: '#2d2d2d'
        }
      }
    },
  },
  plugins: [],
}
