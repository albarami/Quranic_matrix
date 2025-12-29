# QBM Research Platform v2.0

> World's First AI-Powered Quranic Behavioral Research Platform

![QBM Platform](https://via.placeholder.com/1200x400/065f46/ffffff?text=Quranic+Behavioral+Matrix)

## 🌟 Overview

The **Quranic Behavioral Matrix (QBM)** Research Platform is a cutting-edge tool for exploring behavioral classifications across all 6,236 ayat of the Holy Quran. Powered by **C1 Generative UI**, it provides:

- 🔍 **Natural Language Research** — Ask questions, get interactive visualizations
- 🌐 **Quran Explorer** — Visual browser with coverage heatmaps
- 📊 **AI Dashboard** — Auto-generated statistics and charts
- 💡 **Insights Discovery** — AI-uncovered patterns and correlations

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Generative UI** | AI creates charts, tables, and cards from natural language |
| **15,847 Annotations** | Complete behavioral classification dataset |
| **5 Tafsir Sources** | Ibn Kathir, Tabari, Qurtubi, Sa'di, Jalalayn |
| **87 Behavior Concepts** | Based on Bouzidani's five-context framework |
| **Arabic Support** | Full RTL, tashkeel, and classical fonts |
| **Real-time Streaming** | Responses render progressively |

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone https://github.com/YOUR_USERNAME/qbm-frontend.git
cd qbm-frontend
npm install

# 2. Configure
cp .env.example .env.local
# Edit .env.local with your THESYS_API_KEY

# 3. Run
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
src/
├── app/
│   ├── page.tsx              # Stunning home page
│   ├── (research)/
│   │   └── research/page.tsx # C1Chat research assistant
│   ├── explorer/page.tsx     # Visual Quran browser
│   ├── dashboard/page.tsx    # AI-generated dashboard
│   ├── insights/page.tsx     # Discovered patterns
│   ├── api/chat/route.ts     # C1 API with tools
│   └── components/
│       └── Navigation.tsx
├── lib/                       # Utilities
└── globals.css               # Islamic design system
```

## 🎨 Pages

### Home (`/`)
- Animated hero with featured verse
- Live statistics counters
- AI-discovered insights cards
- Interactive demo previews

### Research (`/research`)
- C1Chat interface with example queries
- Categories: Behavioral, Tafsir, Statistical, Cross-Reference
- Welcome screen with capability showcase

### Explorer (`/explorer`)
- 114 surah grid with coverage heatmap
- Click-to-drill-down interface
- Filter by behavior type
- AI-generated surah analysis

### Dashboard (`/dashboard`)
- Auto-generating statistics sections
- Agent distribution pie chart
- Behavior breakdown bar charts
- Recent activity table

### Insights (`/insights`)
- AI-discovered research patterns
- Interactive detail panels
- Export capabilities

## 🛠 Technology

| Layer | Technology |
|-------|------------|
| Framework | Next.js 14 |
| Generative UI | C1/Thesys SDK |
| Styling | Tailwind CSS |
| Animation | Framer Motion |
| Charts | Recharts |
| Fonts | Inter, Amiri, Scheherazade |

## 🔧 Configuration

### Environment Variables

```env
THESYS_API_KEY=your_api_key     # Required
C1_MODEL=c1-nightly              # Optional
QBM_BACKEND_URL=http://...       # Optional
```

### Getting a Thesys API Key

1. Go to [console.thesys.dev](https://console.thesys.dev)
2. Sign up / Sign in
3. Navigate to "API Keys"
4. Create a new key
5. Copy to `.env.local`

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Ayat | 6,236 |
| Behavioral Spans | 15,847 |
| Gold Tier | 4,231 |
| Silver Tier | 7,892 |
| Average IAA | 0.78 |

## 🤝 Contributing

This is a research project. Contact Salim Al-Barami for collaboration.

## 📄 Citation

```bibtex
@misc{qbm2025,
  title={Quranic Behavioral Matrix: AI-Powered Research Platform},
  author={Al-Barami, Salim},
  year={2025}
}
```

## 📜 License

Research use only. Contact for permissions.

---

*Built for Islamic scholarship with ❤️*
