/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            boxShadow: {
                glow: "0 0 25px rgba(155, 99, 255, 0.6)",
            },
            colors: {
                moonlightPurple: "#9b63ff",
                moonlightBlue: "#3b4cca",
            },
        },
    },
    plugins: [],
}
