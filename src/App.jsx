import { useMemo, useState } from 'react';
import {
  ArrowUpRight, BookOpen, ChevronDown, ChevronUp, CircleAlert,
  GitCompareArrows, LoaderCircle, Search, Sparkles, X,
} from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const METHODS = [
  { value: 'hybrid', label: 'Hybrid', detail: 'RRF fusion' },
  { value: 'dense', label: 'Dense', detail: 'Vector' },
  { value: 'sparse', label: 'Sparse', detail: 'BM25' },
];
const PRESETS = [
  'LoRA parameter-efficient fine-tuning',
  'FlashAttention GPU optimization',
  'Vision Transformer patch projection',
];

async function searchPapers(query, method, topK, signal) {
  const response = await fetch(`${API_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, method, top_k: topK }),
    signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail || `Search failed (${response.status})`);
  }
  return response.json();
}

function ResultCard({ paper, index }) {
  const [expanded, setExpanded] = useState(false);
  const categories = (paper.categories || '').split(',').map((item) => item.trim()).filter(Boolean);
  return (
    <article className="result-card group">
      <div className="flex items-start gap-4">
        <span className="rank-badge">{String(index + 1).padStart(2, '0')}</span>
        <div className="min-w-0 flex-1">
          <a className="result-title" href={paper.url} target="_blank" rel="noreferrer">
            {paper.title}<ArrowUpRight size={16} className="inline-block ml-1 opacity-50 group-hover:opacity-100" />
          </a>
          <p className="mt-2 text-sm text-slate-500">{paper.authors || 'Authors unavailable'} <span className="px-1.5 text-slate-300">·</span>{paper.published?.slice(0, 10) || 'Date unavailable'}</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {categories.map((category) => <span className="category-pill" key={category}>{category}</span>)}
          </div>
          <div className="mt-4">
            <p className={`abstract ${expanded ? '' : 'line-clamp-3'}`}>{paper.abstract || 'No abstract available.'}</p>
            <button className="expand-button" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>
              {expanded ? <>Show less <ChevronUp size={14} /></> : <>Read abstract <ChevronDown size={14} /></>}
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function ResultsList({ title, subtitle, papers, loading }) {
  return (
    <section className="min-w-0">
      <div className="mb-4 flex items-end justify-between gap-3">
        <div><h2 className="text-lg font-semibold text-slate-900">{title}</h2><p className="mt-0.5 text-xs text-slate-500">{subtitle}</p></div>
        {!loading && <span className="count-pill">{papers.length} papers</span>}
      </div>
      {loading ? <div className="space-y-3" aria-label="Loading results">{[0, 1, 2].map((item) => <div className="skeleton-card" key={item}><div className="skeleton-line w-3/4" /><div className="skeleton-line w-2/5" /><div className="skeleton-line w-full" /></div>)}</div>
        : papers.length ? <div className="space-y-3">{papers.map((paper, index) => <ResultCard key={`${paper.url}-${index}`} paper={paper} index={index} />)}</div>
          : <div className="empty-results"><BookOpen size={22} /><span>No papers found for this search.</span></div>}
    </section>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [method, setMethod] = useState('hybrid');
  const [topK, setTopK] = useState(10);
  const [compare, setCompare] = useState(false);
  const [resultSets, setResultSets] = useState({ primary: [], sparse: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const selectedMethod = useMemo(() => METHODS.find((item) => item.value === method), [method]);

  const runSearch = async (event, override = query, options = {}) => {
    event?.preventDefault();
    const cleanQuery = override.trim();
    if (!cleanQuery) return;
    setQuery(cleanQuery);
    setSubmittedQuery(cleanQuery);
    setLoading(true);
    setError('');
    const controller = new AbortController();
    const activeMethod = options.method ?? method;
    const activeTopK = options.topK ?? topK;
    const activeCompare = options.compare ?? compare;
    try {
      const requests = activeCompare
        ? [searchPapers(cleanQuery, 'hybrid', activeTopK, controller.signal), searchPapers(cleanQuery, 'sparse', activeTopK, controller.signal)]
        : [searchPapers(cleanQuery, activeMethod, activeTopK, controller.signal)];
      const responses = await Promise.all(requests);
      setResultSets({ primary: responses[0].results || [], sparse: activeCompare ? responses[1].results || [] : [] });
    } catch (cause) {
      setResultSets({ primary: [], sparse: [] });
      setError(cause.name === 'TypeError' ? 'Could not reach the search service. Check that the API is running and try again.' : cause.message);
    } finally {
      setLoading(false);
    }
  };

  const clearSearch = () => { setQuery(''); setSubmittedQuery(''); setResultSets({ primary: [], sparse: [] }); setError(''); };

  return (
    <main className="min-h-screen bg-[#f5f7fb] text-slate-900">
      <header className="border-b border-slate-200/80 bg-white/90">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
          <a href="#top" className="flex items-center gap-3" aria-label="ArXivLens home">
            <span className="brand-mark"><Sparkles size={20} /></span>
            <span><span className="block text-lg font-bold tracking-tight">ArXiv<span className="text-indigo-600">Lens</span></span><span className="hidden text-[11px] text-slate-500 sm:block">Research, in focus.</span></span>
          </a>
          <span className="hidden rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-500 md:inline-flex">ML · AI · NLP</span>
        </div>
      </header>

      <div id="top" className="mx-auto max-w-7xl px-5 pb-16 pt-10 sm:px-8 sm:pt-14">
        <section className="mb-8">
          <p className="eyebrow"><span className="h-1.5 w-1.5 rounded-full bg-indigo-500" /> ARXIV RESEARCH DISCOVERY</p>
          <h1 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight text-slate-950 sm:text-5xl">Find the ideas behind <span className="text-indigo-600">the paper.</span></h1>
          <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">Hybrid Semantic Search &amp; Retrieval Benchmarking for AI Research</p>
          <div className="mt-5 flex flex-wrap gap-2"><span className="status-badge"><span className="status-dot bg-violet-500" />FAISS Dense</span><span className="status-badge"><span className="status-dot bg-sky-500" />BM25 Sparse</span><span className="status-badge"><span className="status-dot bg-emerald-500" />RRF Fusion</span></div>
        </section>

        <section className="search-panel">
          <form onSubmit={runSearch}>
            <label className="sr-only" htmlFor="paper-query">Search papers</label>
            <div className="search-input-wrap"><Search size={19} className="shrink-0 text-slate-400" /><input id="paper-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Describe a topic, method, or research question..." /><button type="button" onClick={clearSearch} disabled={!query} className="clear-button" aria-label="Clear search"><X size={17} /></button><button className="search-button" type="submit" disabled={!query.trim() || loading}>{loading ? <LoaderCircle size={17} className="animate-spin" /> : <Search size={17} />}<span>Search</span></button></div>
          </form>

          <div className="mt-5 flex flex-col justify-between gap-5 lg:flex-row lg:items-center">
              <div className="flex flex-wrap items-center gap-4">
              <div className="segmented-control" role="radiogroup" aria-label="Retrieval method">{METHODS.map((item) => <button key={item.value} type="button" role="radio" aria-checked={method === item.value} onClick={() => { setMethod(item.value); if (submittedQuery && !compare) runSearch(null, submittedQuery, { method: item.value }); }} className={`method-option ${method === item.value ? 'selected' : ''}`}><span>{item.label}</span><small>{item.detail}</small></button>)}</div>
              <label className="flex items-center gap-2 text-xs font-medium text-slate-600">Top K<select className="topk-select" value={topK} onChange={(event) => { const value = Number(event.target.value); setTopK(value); if (submittedQuery) runSearch(null, submittedQuery, { topK: value }); }}><option value={5}>5</option><option value={10}>10</option><option value={20}>20</option></select></label>
            </div>
          <button type="button" className={`compare-toggle ${compare ? 'active' : ''}`} onClick={() => { const next = !compare; setCompare(next); if (submittedQuery) runSearch(null, submittedQuery, { compare: next }); }} aria-pressed={compare}><span className="toggle-track"><span /></span><GitCompareArrows size={15} /><span>Compare Hybrid vs BM25 side-by-side</span></button>
          </div>

          <div className="mt-5 border-t border-slate-100 pt-4"><p className="mb-2.5 text-[11px] font-semibold uppercase tracking-[.12em] text-slate-400">Try a topic</p><div className="flex flex-wrap gap-2">{PRESETS.map((preset) => <button key={preset} type="button" className="preset-chip" onClick={() => runSearch(null, preset)}>{preset}<ArrowUpRight size={12} /></button>)}</div></div>
        </section>

        {error && <div role="alert" className="error-banner"><CircleAlert size={18} /><div><strong>Search unavailable</strong><p>{error}</p></div><button onClick={() => runSearch(null, submittedQuery || query)} className="ml-auto underline underline-offset-2">Retry</button></div>}

        <section className="mt-9">
          {!submittedQuery && !loading ? <div className="welcome-state"><span className="welcome-icon"><Search size={22} /></span><h2>What are you researching?</h2><p>Search across AI and machine learning papers with keywords or natural language.</p></div>
            : <><div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="eyebrow">SEARCH RESULTS</p><h2 className="mt-1 text-xl font-semibold">{submittedQuery ? <>Results for <span className="text-slate-500">“{submittedQuery}”</span></> : 'Searching...'}</h2></div>{submittedQuery && <span className="count-pill">{selectedMethod?.label} · Top {topK}</span>}</div>
              <div className={`grid gap-6 ${compare ? 'lg:grid-cols-2' : 'grid-cols-1'}`}><ResultsList title={compare ? 'Hybrid Search' : `${selectedMethod?.label} Results`} subtitle={compare ? 'Dense + sparse fused with reciprocal rank' : `${selectedMethod?.detail} retrieval`} papers={resultSets.primary} loading={loading} />{compare && <ResultsList title="BM25 Keyword Search" subtitle="Sparse lexical matching" papers={resultSets.sparse} loading={loading} />}</div></>}
        </section>
        <footer className="mt-14 border-t border-slate-200 pt-5 text-center text-xs text-slate-400">Discover connections across machine learning research.</footer>
      </div>
    </main>
  );
}
