``` mermaid
flowchart LR
    A[Collect\nCKAN: Événements publics] --> B[Clean (minimal)\nadd aliases; keep FR fields]
    B --> C[Hard Filters\nwindow=N days; audience; boroughs; types; no 'en ligne']
    C --> D[Embedding Ranker\nOllama nomic-embed-text\ncosine vs. likes]
    D -->|top K| E[Weather Enrichment\nOpen-Meteo: temp & rain]
    E --> F[LLM Selection\nGroq: choose final N]
    F --> G[LLM Newsletter\nTop Picks / Free / Outdoor\nEnglish Markdown]
    G --> H[Publish\ncommit reports/weekly_tldr.md\nupload artifact]
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
```