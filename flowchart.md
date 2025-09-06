
flowchart LR
    A[Collect<br/>CKAN: Événements publics] --> B[Clean (minimal)<br/>add aliases; keep FR fields]
    B --> C[Hard Filters<br/>window=N days; audience; boroughs; types; no 'en ligne']
    C --> D[Embedding Ranker<br/>Ollama nomic-embed-text<br/>cosine vs. likes]
    D -->|top K| E[Weather Enrichment<br/>Open-Meteo: temp & rain]
    E --> F[LLM Selection<br/>Groq: choose final N]
    F --> G[LLM Newsletter<br/>Top Picks / Free / Outdoor<br/>English Markdown]
    G --> H[Publish<br/>commit reports/weekly_tldr.md<br/>upload artifact]

    subgraph "GitHub Actions (Thu night schedule)"
      A
      B
      C
      D
      E
      F
      G
      H
    end
