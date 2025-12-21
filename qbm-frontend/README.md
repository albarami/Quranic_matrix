# QBM Frontend - Quranic Behavioral Matrix Research Interface

> A Generative UI frontend for exploring the Quranic Behavioral Matrix using C1/Thesys

![QBM Logo](https://via.placeholder.com/800x200/065f46/ffffff?text=QBM+Research+Interface)

## Overview

This is the frontend application for the **Quranic Behavioral Matrix (QBM)** project. It provides:

- 🔍 **Research Assistant** — Natural language queries about Quranic behaviors
- 📊 **Dashboard** — Project statistics, coverage metrics, and progress tracking
- ✏️ **Annotator Workbench** — Tafsir-integrated annotation interface

Built with [Next.js](https://nextjs.org/), [C1/Thesys](https://thesys.dev/), and [Tailwind CSS](https://tailwindcss.com/).

---

## Features

| Feature | Description |
|---------|-------------|
| **Generative UI** | AI generates interactive tables, charts, and forms based on queries |
| **Tafsir Integration** | Side-by-side tafsir comparison (Ibn Kathir, Tabari, Qurtubi, Sa'di, Jalalayn) |
| **Arabic Support** | RTL layout, Arabic fonts (Amiri), proper tashkeel display |
| **Real-time Streaming** | Responses stream as they're generated |
| **Tool Calling** | Backend integration for live data queries |

---

## Prerequisites

- **Node.js** 18.x or later
- **npm** or **yarn**
- **Thesys API Key** — Get one at [console.thesys.dev](https://console.thesys.dev/keys)
- **QBM Backend** (optional) — FastAPI backend for live data

---

## Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qbm-frontend.git
cd qbm-frontend

# Install dependencies
npm install
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env.local

# Edit with your values
nano .env.local
```

Required variables:

```env
THESYS_API_KEY=your_thesys_api_key
QBM_BACKEND_URL=http://localhost:8000  # Your FastAPI backend
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Project Structure

```
qbm-frontend/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── chat/
│   │   │       └── route.ts      # C1 API endpoint with QBM tools
│   │   ├── research/
│   │   │   └── page.tsx          # Research Assistant (C1Chat)
│   │   ├── dashboard/
│   │   │   └── page.tsx          # Project Dashboard (C1Component)
│   │   ├── annotate/
│   │   │   └── page.tsx          # Annotator Workbench
│   │   ├── components/
│   │   │   └── Navigation.tsx    # Main navigation
│   │   ├── layout.tsx            # Root layout
│   │   ├── page.tsx              # Home page
│   │   └── globals.css           # Global styles + Arabic fonts
│   └── lib/
│       └── qbm-tools.ts          # Tool definitions & execution
├── public/                        # Static assets
├── .env.example                   # Environment template
├── next.config.js                 # Next.js configuration
├── tailwind.config.js             # Tailwind configuration
├── tsconfig.json                  # TypeScript configuration
└── package.json                   # Dependencies
```

---

## Pages

### `/` — Home

Landing page with project overview and navigation cards.

### `/research` — Research Assistant

Natural language interface using `<C1Chat>`. Ask questions like:

- "Show me behaviors related to the heart (قلب)"
- "Compare tafsir for Ayat al-Kursi (2:255)"
- "What are the most common behavior concepts?"
- "Find speech acts in Surah Al-Hujurat"

### `/dashboard` — Project Dashboard

Auto-generated statistics using `<C1Component>`:

- Annotation progress (pie chart)
- Coverage by surah (bar chart)
- Quality metrics (IAA scores)
- Recent activity table

### `/annotate` — Annotator Workbench

Annotation interface with:

- Ayah display with Arabic text
- Tafsir panels (tabbed)
- Annotation form
- Quick navigation sidebar

---

## C1 Tools

The frontend defines these tools for C1 to call:

| Tool | Description |
|------|-------------|
| `search_spans` | Search behavioral annotations by concept, surah, agent, organ |
| `get_tafsir` | Retrieve tafsir for a specific ayah |
| `get_statistics` | Get dataset statistics and coverage metrics |
| `compare_tafsir` | Compare interpretations across tafsir sources |
| `get_ayah` | Get full ayah text and metadata |

Tools are defined in `src/lib/qbm-tools.ts`. In production, update the implementations to call your FastAPI backend.

---

## Backend Integration

The frontend expects a FastAPI backend at `QBM_BACKEND_URL` with these endpoints:

```
POST /api/spans/search     # Search annotations
GET  /api/tafsir/{surah}/{ayah}
GET  /api/tafsir/compare/{surah}/{ayah}
GET  /api/statistics
GET  /api/ayah/{surah}/{ayah}
```

For development, the tools return mock data. Update `src/lib/qbm-tools.ts` to connect to your real backend.

---

## Customization

### Theming

Edit `tailwind.config.js` to customize colors:

```javascript
colors: {
  qbm: {
    primary: '#065f46',    // Emerald 800
    secondary: '#059669',  // Emerald 600
    accent: '#10b981',     // Emerald 500
  },
}
```

### Arabic Fonts

The project uses:
- **Amiri** — For Quranic text
- **Inter** — For UI elements

Add more fonts in `src/app/globals.css`.

### C1 Model

Change the model in `.env.local`:

```env
C1_MODEL=c1/anthropic/claude-sonnet-4/v-20250617
```

---

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables in Vercel dashboard
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

---

## Development

### Adding New Tools

1. Define the tool in `src/lib/qbm-tools.ts`:

```typescript
{
  type: "function",
  function: {
    name: "my_new_tool",
    description: "...",
    parameters: { ... }
  }
}
```

2. Implement the execution in `executeTools()`:

```typescript
case "my_new_tool":
  result = await myNewToolFunction(args);
  break;
```

### Adding New Pages

1. Create `src/app/my-page/page.tsx`
2. Add to navigation in `src/app/components/Navigation.tsx`

---

## Troubleshooting

### "Failed to fetch" errors

- Check that `THESYS_API_KEY` is set correctly
- Verify the API key is valid at console.thesys.dev

### Arabic text not displaying correctly

- Ensure Amiri font is loaded (check Network tab)
- Use `dir="rtl"` for Arabic containers
- Use `font-arabic` Tailwind class

### Tools returning mock data

- Update tool implementations in `src/lib/qbm-tools.ts`
- Connect to your FastAPI backend

---

## Related Projects

- [qbm-backend](../backend) — FastAPI backend with PostgreSQL
- [QBM Project Plan](../PROJECT_PLAN.md) — Full project documentation
- [Coding Manual](../docs/coding_manual_v1.docx) — Annotation guidelines

---

## License

This project is part of the Quranic Behavioral Matrix research initiative.  
Contact for permissions.

---

## Citation

```bibtex
@misc{qbm2025,
  title={Quranic Behavioral Matrix: A Structured Dataset of Quranic Behavioral Classifications},
  author={Al-Barami, Salim and Bouzidani, Ibrahim},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/quranic-behavior-matrix}
}
```

---

## Contributors

- **Salim Al-Barami** — Project Lead
- **Dr. Ibrahim Bouzidani** — Framework Design

---

*Built with ❤️ for Islamic scholarship*
