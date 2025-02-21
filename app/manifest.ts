import { MetadataRoute } from 'next'
 
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Expense Sorted App',
    short_name: 'Expense Sorted App',
    description: 'Automatically categorise your monthly expenses using AI. Hook this App up to your Google Sheets™ and get your monthly budgeting done in no time.',
    start_url: '/',
    display: 'standalone',
    background_color: '#1E1E1E',
    theme_color: '#9C59EE',
    icons: [
      {
        src: '/favicon.ico',
        sizes: 'any',
        type: 'image/x-icon',
      },
    ],
  }
}