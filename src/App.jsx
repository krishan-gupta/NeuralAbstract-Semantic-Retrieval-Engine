import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowUpRight, BookOpen, Check, ChevronDown, ChevronUp, CircleAlert,
  FlaskConical, GitCompareArrows, LoaderCircle, Plus, Search, Sparkles, Tag, X,
  Undo2, SkipForward, Download, Layers3,
} from 'lucide-react';

// Match Uvicorn's explicit IPv4 bind address. Some Windows setups resolve
// `localhost` to IPv6 first, while the API listens only on 127.0.0.1.
const API_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
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

async function api(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function ResultCard({ paper, index, uniqueLabel }) {
  const [expanded, setExpanded] = useState(false);
  const categories = (paper.categories || '').split(/[\s,]+/).map((item) => item.trim()).filter(Boolean);
  return (
    <article className={`result-card group ${paper.unique ? 'unique-card' : ''}`}>
      <div className="flex items-start gap-4">
        <span className="rank-badge">{String(index + 1).padStart(2, '0')}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <a className="result-title" href={paper.url} target="_blank" rel="noreferrer">
              {paper.title}<ArrowUpRight size={16} className="ml-1 inline-block opacity-50 group-hover:opacity-100" />
            </a>
            {paper.unique && <span className={`unique-badge ${uniqueLabel === 'Only in Hybrid' ? 'hybrid-unique' : 'bm25-unique'}`}>{uniqueLabel}</span>}
          </div>
          <p className="mt-2 text-sm text-slate-500">{paper.authors || 'Authors unavailable'} <span className="px-1.5 text-slate-300">·</span>{paper.published?.slice(0, 10) || 'Date unavailable'}</p>
          <div className="mt-3 flex flex-wrap gap-1.5">{categories.map((category) => <span className="category-pill" key={category}>{category}</span>)}</div>
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

function ResultsList({ title, subtitle, papers, loading, uniqueLabel }) {
  return (
    <section className="min-w-0">
      <div className="mb-4 flex items-end justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-900">{title}</h2><p className="mt-0.5 text-xs text-slate-500">{subtitle}</p></div>{!loading && <span className="count-pill">{papers.length} papers</span>}</div>
      {loading ? <div className="space-y-3" aria-label="Loading results">{[0, 1, 2].map((item) => <div className="skeleton-card" key={item}><div className="skeleton-line w-3/4" /><div className="skeleton-line w-2/5" /><div className="skeleton-line w-full" /></div>)}</div>
        : papers.length ? <div className="space-y-3">{papers.map((paper, index) => <ResultCard key={`${paper.url}-${index}`} paper={paper} index={index} uniqueLabel={uniqueLabel} />)}</div>
          : <div className="empty-results"><BookOpen size={22} /><span>No papers found for this search.</span></div>}
    </section>
  );
}

function LabelCard({ paper, onLabel, saving, active, index }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <article className={`label-card ${active ? 'ring-2 ring-indigo-400 ring-offset-2' : ''}`} aria-current={active ? 'true' : undefined}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1"><span className="rank-badge mr-2">{String(index + 1).padStart(2, '0')}</span><a className="result-title" href={paper.url} target="_blank" rel="noreferrer">{paper.title}<ArrowUpRight size={15} className="ml-1 inline" /></a><p className="mt-1 text-xs text-slate-500">{paper.authors} · {paper.published?.slice(0, 10)}</p></div>
        <div className="flex gap-2"><button disabled={saving} className={`judgment-button relevant ${paper.labeled && paper.relevant ? 'chosen' : ''}`} onClick={() => onLabel(paper, true)}><Check size={14} />Relevant</button><button disabled={saving} className={`judgment-button irrelevant ${paper.labeled && !paper.relevant ? 'chosen' : ''}`} onClick={() => onLabel(paper, false)}><X size={14} />Not relevant</button></div>
      </div>
      <p className={`mt-3 text-sm leading-6 text-slate-600 ${expanded ? '' : 'line-clamp-3'}`}>{paper.abstract}</p>
      <button className="expand-button" onClick={() => setExpanded(!expanded)}>{expanded ? 'Show less' : 'Read abstract'}</button>
    </article>
  );
}

function nextUnlabeledIndex(papers, currentIndex, excluded = new Set()) {
  if (!papers.length) return -1;
  if (currentIndex < 0) {
    return papers.findIndex((paper) => !paper.labeled && !excluded.has(paper.arxiv_id));
  }
  for (let offset = 1; offset < papers.length; offset += 1) {
    const index = (currentIndex + offset) % papers.length;
    if (!papers[index].labeled && !excluded.has(papers[index].arxiv_id)) return index;
  }
  return -1;
}

function MetricBar({ label, value }) {
  return <div className="metric-bar-row"><span>{label}</span><div className="metric-bar-track"><div className="metric-bar-fill" style={{ width: `${Math.max(0, Math.min(100, Number(value || 0) * 100))}%` }} /></div><strong>{(Number(value || 0) * 100).toFixed(1)}%</strong></div>;
}

export default function App() {
  const [tab, setTab] = useState('search');
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [method, setMethod] = useState('hybrid');
  const [embedder, setEmbedder] = useState('local');
  const [embedderOptions, setEmbedderOptions] = useState(['local']);
  const [topK, setTopK] = useState(10);
  const [compare, setCompare] = useState(false);
  const [resultSets, setResultSets] = useState({ primary: [], sparse: [] });
  const [overlap, setOverlap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [queries, setQueries] = useState([]);
  const [activeQueryId, setActiveQueryId] = useState(null);
  const [queryText, setQueryText] = useState('');
  const [pool, setPool] = useState([]);
  const [activePaperIndex, setActivePaperIndex] = useState(0);
  const [skippedPaperIds, setSkippedPaperIds] = useState([]);
  const [lastLabelAction, setLastLabelAction] = useState(null);
  const [labelProgress, setLabelProgress] = useState({ labeled: 0, total: 0 });
  const [labelLoading, setLabelLoading] = useState(false);
  const [labelSaving, setLabelSaving] = useState(false);
  const [labelError, setLabelError] = useState('');
  const [labelNotice, setLabelNotice] = useState('');
  const [poolBuilding, setPoolBuilding] = useState(false);
  const [exportMessage, setExportMessage] = useState('');

  const [benchmark, setBenchmark] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState('');
  const selectedMethod = useMemo(() => METHODS.find((item) => item.value === method), [method]);

  useEffect(() => {
    api('/embedders').then((data) => {
      const available = data.embedders?.length ? data.embedders : ['local'];
      setEmbedderOptions(available);
      if (!available.includes(embedder)) setEmbedder(available[0]);
    }).catch(() => {});
  }, []);

  const runSearch = async (event, override = query, options = {}) => {
    event?.preventDefault();
    const cleanQuery = override.trim();
    if (!cleanQuery) return;
    setQuery(cleanQuery);
    setSubmittedQuery(cleanQuery);
    setLoading(true);
    setError('');
    const activeMethod = options.method ?? method;
    const activeTopK = options.topK ?? topK;
    const activeCompare = options.compare ?? compare;
    const activeEmbedder = options.embedder ?? embedder;
    try {
      if (activeCompare) {
        const data = await api('/compare', { method: 'POST', body: JSON.stringify({ query: cleanQuery, top_k: activeTopK, embedder: activeEmbedder }) });
        setResultSets({ primary: data.hybrid || [], sparse: data.bm25 || [] });
        setOverlap(data.overlap);
      } else {
        const data = await api('/search', { method: 'POST', body: JSON.stringify({ query: cleanQuery, method: activeMethod, top_k: activeTopK, embedder: activeEmbedder }) });
        setResultSets({ primary: data.results || [], sparse: [] });
        setOverlap(null);
      }
    } catch (cause) {
      setResultSets({ primary: [], sparse: [] });
      setOverlap(null);
      setError(cause instanceof TypeError ? 'Could not reach the API. Confirm it is running at the configured address.' : cause.message);
    } finally { setLoading(false); }
  };

  const clearSearch = () => { setQuery(''); setSubmittedQuery(''); setResultSets({ primary: [], sparse: [] }); setOverlap(null); setError(''); };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const initialTab = params.get('tab');
    if (initialTab === 'benchmark' || initialTab === 'label') setTab(initialTab);
    const initialQuery = params.get('q')?.trim();
    if (!initialQuery) return;
    const initialCompare = params.get('compare') === '1';
    const initialMethod = METHODS.some((item) => item.value === params.get('method')) ? params.get('method') : 'hybrid';
    const initialEmbedder = params.get('embedder') || 'local';
    setCompare(initialCompare);
    setMethod(initialMethod);
    setEmbedder(initialEmbedder);
    runSearch(null, initialQuery, { compare: initialCompare, method: initialMethod, embedder: initialEmbedder });
  }, []);

  const loadQueries = useCallback(async () => {
    try {
      const data = await api('/benchmark/queries');
      const available = data.queries || [];
      setQueries(available);
      setActiveQueryId((current) => current || available[0]?.id || null);
      return available;
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); return []; }
  }, []);

  useEffect(() => { if (tab === 'label') loadQueries(); }, [tab, loadQueries]);

  const loadPool = useCallback(async (queryId) => {
    const data = await api(`/labels/pool?query_id=${queryId}`);
    const papers = data.papers || [];
    setQueryText(data.query);
    setPool(papers);
    setSkippedPaperIds([]);
    setLastLabelAction(null);
    setActivePaperIndex(nextUnlabeledIndex(papers, -1));
    setLabelProgress({ labeled: data.labeled_count || 0, total: data.pool_count || 0 });
  }, []);

  useEffect(() => {
    if (tab !== 'label' || !activeQueryId) return;
    let cancelled = false;
    setLabelLoading(true);
    setLabelError('');
    loadPool(activeQueryId)
      .catch((cause) => { if (!cancelled) setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); })
      .finally(() => { if (!cancelled) setLabelLoading(false); });
    return () => { cancelled = true; };
  }, [tab, activeQueryId, loadPool]);

  useEffect(() => {
    const selected = queries.find((item) => item.id === activeQueryId);
    if (selected) setQueryText(selected.text);
  }, [activeQueryId, queries]);

  const saveJudgment = async (paper, relevant) => {
    if (!activeQueryId || labelSaving) return;
    setLabelSaving(true);
    setLabelError('');
    setLabelNotice('');
    const queryId = activeQueryId;
    const previousRelevant = paper.labeled ? Boolean(paper.relevant) : null;
    try {
      await api('/labels', { method: 'POST', body: JSON.stringify({ query_id: queryId, arxiv_id: paper.arxiv_id, relevant }) });
      const updatedPool = pool.map((item) => item.arxiv_id === paper.arxiv_id ? { ...item, labeled: true, relevant } : item);
      setPool(updatedPool);
      const remainingSkipped = skippedPaperIds.filter((arxivId) => arxivId !== paper.arxiv_id);
      setSkippedPaperIds(remainingSkipped);
      setLastLabelAction({ type: 'label', queryId, arxivId: paper.arxiv_id, previousRelevant, nextRelevant: relevant, index: pool.findIndex((item) => item.arxiv_id === paper.arxiv_id) });
      const labeledDelta = previousRelevant === null ? 1 : 0;
      const relevantDelta = Number(relevant) - Number(previousRelevant || false);
      setLabelProgress((progress) => ({ ...progress, labeled: progress.labeled + labeledDelta }));
      setQueries((items) => items.map((item) => item.id === queryId ? { ...item, labeled_count: Number(item.labeled_count || 0) + labeledDelta, relevant_count: Number(item.relevant_count || 0) + relevantDelta } : item));
      setActivePaperIndex(nextUnlabeledIndex(updatedPool, pool.findIndex((item) => item.arxiv_id === paper.arxiv_id), new Set(remainingSkipped)));
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
    finally { setLabelSaving(false); }
  };

  const skipCurrentPaper = () => {
    if (!pool.length || activePaperIndex < 0) return;
    const paper = pool[activePaperIndex];
    const nextSkipped = [...new Set([...skippedPaperIds, paper.arxiv_id])];
    setLastLabelAction({ type: 'skip', index: activePaperIndex, arxivId: paper.arxiv_id });
    setSkippedPaperIds(nextSkipped);
    setActivePaperIndex(nextUnlabeledIndex(pool, activePaperIndex, new Set(nextSkipped)));
  };

  const undoLastAction = async () => {
    if (!lastLabelAction || labelSaving) return;
    if (lastLabelAction.type === 'skip') {
      setSkippedPaperIds((items) => items.filter((arxivId) => arxivId !== lastLabelAction.arxivId));
      setActivePaperIndex(lastLabelAction.index);
      setLastLabelAction(null);
      return;
    }
    setLabelSaving(true);
    setLabelError('');
    try {
      const { queryId, arxivId, previousRelevant, nextRelevant, index } = lastLabelAction;
      if (previousRelevant === null) {
        await api('/labels', { method: 'DELETE', body: JSON.stringify({ query_id: queryId, arxiv_id: arxivId }) });
      } else {
        await api('/labels', { method: 'POST', body: JSON.stringify({ query_id: queryId, arxiv_id: arxivId, relevant: previousRelevant }) });
      }
      const labeledDelta = previousRelevant === null ? -1 : 0;
      const relevantDelta = Number(previousRelevant || false) - Number(nextRelevant);
      if (queryId === activeQueryId) {
        const updatedPool = pool.map((item) => item.arxiv_id === arxivId ? { ...item, labeled: previousRelevant !== null, relevant: previousRelevant } : item);
        setPool(updatedPool);
        setActivePaperIndex(nextUnlabeledIndex(updatedPool, index, new Set(skippedPaperIds)));
        setLabelProgress((progress) => ({ ...progress, labeled: progress.labeled + labeledDelta }));
      }
      setQueries((items) => items.map((item) => item.id === queryId ? { ...item, labeled_count: Number(item.labeled_count || 0) + labeledDelta, relevant_count: Number(item.relevant_count || 0) + relevantDelta } : item));
      setLastLabelAction(null);
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
    finally { setLabelSaving(false); }
  };

  useEffect(() => {
    if (tab !== 'label') return undefined;
    const handleShortcut = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
      const key = event.key.toLowerCase();
      if (key === 'z') { event.preventDefault(); undoLastAction(); return; }
      if (key === 's') { event.preventDefault(); skipCurrentPaper(); return; }
      const paper = pool[activePaperIndex];
      if (paper && !labelSaving && key === 'r') { event.preventDefault(); saveJudgment(paper, true); }
      if (paper && !labelSaving && key === 'n') { event.preventDefault(); saveJudgment(paper, false); }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [tab, pool, activePaperIndex, labelSaving, lastLabelAction, activeQueryId, skippedPaperIds]);

  const refreshAllPools = async () => {
    setPoolBuilding(true);
    setLabelError('');
    try {
      const result = await api('/labels/pool/refresh-all', { method: 'POST', body: '{}' });
      await loadQueries();
      if (activeQueryId) await loadPool(activeQueryId);
      setLabelNotice(`Built pools for ${result.queries_processed} queries; added ${result.papers_added} papers.`);
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
    finally { setPoolBuilding(false); }
  };

  const refreshActivePool = async () => {
    if (!activeQueryId) return;
    setPoolBuilding(true);
    setLabelError('');
    try {
      const result = await api('/labels/pool/refresh', { method: 'POST', body: JSON.stringify({ query_id: activeQueryId }) });
      await loadPool(activeQueryId);
      setQueries((items) => items.map((item) => item.id === activeQueryId ? { ...item, pool_count: result.pool_count } : item));
      setLabelNotice(`Pool refreshed; ${result.added_count} new papers added.`);
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
    finally { setPoolBuilding(false); }
  };

  const exportLabelFile = async () => {
    try {
      const result = await api('/labels/export', { method: 'POST', body: '{}' });
      setExportMessage(`Exported ${result.count} judgments at ${new Date(result.exported_at).toLocaleString()}`);
      setLabelError('');
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
  };

  const goToNextUnlabeledQuery = () => {
    const eligible = queries.filter((item) => Number(item.pool_count || 0) > Number(item.labeled_count || 0));
    if (!eligible.length) return;
    const currentIndex = eligible.findIndex((item) => item.id === activeQueryId);
    const next = eligible[(currentIndex + 1 + eligible.length) % eligible.length];
    setLastLabelAction(null);
    setActiveQueryId(next.id);
  };

  const saveQuery = async () => {
    try {
      const data = await api('/benchmark/queries', { method: 'POST', body: JSON.stringify({ text: queryText, query_id: activeQueryId || undefined }) });
      setActiveQueryId(data.id);
      setLabelError('');
      const available = await loadQueries();
      setQueries(available);
    } catch (cause) { setLabelError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); }
  };

  useEffect(() => {
    if (tab !== 'benchmark') return;
    let cancelled = false;
    setBenchmarkLoading(true);
    setBenchmarkError('');
    api('/benchmark/run', { method: 'POST', body: JSON.stringify({ k_values: [5, 10] }) })
      .then((data) => { if (!cancelled) setBenchmark(data); })
      .catch((cause) => { if (!cancelled) setBenchmarkError(cause instanceof TypeError ? 'Cannot reach the backend API.' : cause.message); })
      .finally(() => { if (!cancelled) setBenchmarkLoading(false); });
    return () => { cancelled = true; };
  }, [tab]);

  const tabs = [{ id: 'search', label: 'Search', icon: Search }, { id: 'label', label: 'Label', icon: Tag }, { id: 'benchmark', label: 'Benchmark', icon: FlaskConical }];
  const readyQueryCount = queries.filter((item) => Number(item.relevant_count || 0) > 0).length;
  const headlineLift = benchmark?.lift_hybrid_vs_bm25?.by_embedder?.['hybrid:local']?.['recall@10'];
  const ndcgComparison = benchmark?.ndcg10_comparison;

  return (
    <main className="min-h-screen bg-[#f5f7fb] text-slate-900">
      <header className="border-b border-slate-200/80 bg-white/90">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
          <a href="#top" className="flex items-center gap-3" aria-label="ArXivLens home"><span className="brand-mark"><Sparkles size={20} /></span><span><span className="block text-lg font-bold tracking-tight">ArXiv<span className="text-indigo-600">Lens</span></span><span className="hidden text-[11px] text-slate-500 sm:block">Research, in focus.</span></span></a>
          <nav className="flex rounded-xl bg-slate-100 p-1" aria-label="Main sections">{tabs.map(({ id, label, icon: Icon }) => <button key={id} className={`app-tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}><Icon size={15} />{label}</button>)}</nav>
        </div>
      </header>

      <div id="top" className="mx-auto max-w-7xl px-5 pb-16 pt-10 sm:px-8 sm:pt-14">
        <section className="mb-8"><p className="eyebrow"><span className="h-1.5 w-1.5 rounded-full bg-indigo-500" /> ARXIV RESEARCH DISCOVERY</p><h1 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight text-slate-950 sm:text-5xl">Find the ideas behind <span className="text-indigo-600">the paper.</span></h1><p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 sm:text-base">Hybrid Semantic Search &amp; Retrieval Benchmarking for AI Research</p><div className="mt-5 flex flex-wrap gap-2"><span className="status-badge"><span className="status-dot bg-violet-500" />FAISS Dense</span><span className="status-badge"><span className="status-dot bg-sky-500" />BM25 Sparse</span><span className="status-badge"><span className="status-dot bg-emerald-500" />RRF Fusion</span></div></section>

        {tab === 'search' && <>
          <section className="search-panel">
            <form onSubmit={runSearch}><label className="sr-only" htmlFor="paper-query">Search papers</label><div className="search-input-wrap"><Search size={19} className="shrink-0 text-slate-400" /><input id="paper-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Describe a topic, method, or research question..." /><button type="button" onClick={clearSearch} disabled={!query} className="clear-button" aria-label="Clear search"><X size={17} /></button><button className="search-button" type="submit" disabled={!query.trim() || loading}>{loading ? <LoaderCircle size={17} className="animate-spin" /> : <Search size={17} />}<span>Search</span></button></div></form>
            <div className="mt-5 flex flex-col justify-between gap-5 lg:flex-row lg:items-center"><div className="flex flex-wrap items-center gap-4"><div className="segmented-control" role="radiogroup" aria-label="Retrieval method">{METHODS.map((item) => <button key={item.value} type="button" role="radio" aria-checked={method === item.value} onClick={() => { setMethod(item.value); if (submittedQuery && !compare) runSearch(null, submittedQuery, { method: item.value }); }} className={`method-option ${method === item.value ? 'selected' : ''}`}><span>{item.label}</span><small>{item.detail}</small></button>)}</div><label className="flex items-center gap-2 text-xs font-medium text-slate-600">Top K<select className="topk-select" value={topK} onChange={(event) => { const value = Number(event.target.value); setTopK(value); if (submittedQuery) runSearch(null, submittedQuery, { topK: value }); }}><option value={5}>5</option><option value={10}>10</option><option value={20}>20</option></select></label>{embedderOptions.length > 1 && <label className="flex items-center gap-2 text-xs font-medium text-slate-600">Embedder<select className="topk-select" value={embedder} onChange={(event) => { setEmbedder(event.target.value); if (submittedQuery) runSearch(null, submittedQuery, { embedder: event.target.value }); }}>{embedderOptions.map((option) => <option value={option} key={option}>{option === 'local' ? 'MiniLM' : option === 'bge' ? 'BGE Small' : option}</option>)}</select></label>}</div><button type="button" className={`compare-toggle ${compare ? 'active' : ''}`} onClick={() => { const next = !compare; setCompare(next); if (submittedQuery) runSearch(null, submittedQuery, { compare: next }); }} aria-pressed={compare}><span className="toggle-track"><span /></span><GitCompareArrows size={15} /><span>Compare Hybrid vs BM25 side-by-side</span></button></div>
            <div className="mt-5 border-t border-slate-100 pt-4"><p className="mb-2.5 text-[11px] font-semibold uppercase tracking-[.12em] text-slate-400">Try a topic</p><div className="flex flex-wrap gap-2">{PRESETS.map((preset) => <button key={preset} type="button" className="preset-chip" onClick={() => runSearch(null, preset)}>{preset}<ArrowUpRight size={12} /></button>)}</div></div>
          </section>
          {error && <div role="alert" className="error-banner"><CircleAlert size={18} /><div><strong>Search unavailable</strong><p>{error}</p></div><button onClick={() => runSearch(null, submittedQuery || query)} className="ml-auto underline underline-offset-2">Retry</button></div>}
          <section className="mt-9">{!submittedQuery && !loading ? <div className="welcome-state"><span className="welcome-icon"><Search size={22} /></span><h2>What are you researching?</h2><p>Search across AI and machine learning papers with keywords or natural language.</p></div> : <><div className="mb-5 flex flex-wrap items-center justify-between gap-3"><div><p className="eyebrow">SEARCH RESULTS</p><h2 className="mt-1 text-xl font-semibold">Results for <span className="text-slate-500">“{submittedQuery}”</span></h2></div>{submittedQuery && <span className="count-pill">{compare ? 'Hybrid + BM25' : `${selectedMethod?.label} · ${embedder}`} · Top {topK}</span>}</div>
            {compare && overlap && <div className="overlap-banner"><div className="overlap-copy"><div className="flex items-center gap-2"><strong>{overlap.overlap_pct}% overlap</strong><span className="overlap-tip" title="Overlap measures how much the two methods agree. It is not accuracy." aria-label="Overlap measures agreement, not accuracy">i</span></div><p>{overlap.shared_count} shared · {overlap.hybrid_only_count} found only by Hybrid · {overlap.bm25_only_count} only by BM25 <span className="agreement-note">Agreement, not accuracy.</span></p></div><div className="overlap-track"><div className="overlap-fill" style={{ width: `${overlap.overlap_pct}%` }} /></div></div>}
            <div className={`grid gap-6 ${compare ? 'lg:grid-cols-2' : 'grid-cols-1'}`}><ResultsList title={compare ? 'Hybrid Search' : `${selectedMethod?.label} Results`} subtitle={compare ? 'Dense + sparse fused with reciprocal rank' : `${selectedMethod?.detail} retrieval`} papers={resultSets.primary} loading={loading} uniqueLabel="Only in Hybrid" />{compare && <ResultsList title="BM25 Keyword Search" subtitle="Sparse lexical matching" papers={resultSets.sparse} loading={loading} uniqueLabel="Only in BM25" />}</div></>}</section>
        </>}

        {tab === 'label' && <section className="content-panel">
          <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
            <div><p className="eyebrow"><Tag size={13} /> HUMAN RELEVANCE LABELS</p><h2 className="mt-2 text-2xl font-semibold">Build the benchmark together</h2><p className="mt-1 text-sm text-slate-500">Judge papers without seeing which search method found them.</p></div>
            <div className="flex flex-wrap gap-2">
              <button className="preset-chip" onClick={refreshActivePool} disabled={!activeQueryId || poolBuilding}><Layers3 size={14} />Refresh this pool</button>
              <button className="search-button" onClick={refreshAllPools} disabled={poolBuilding}>{poolBuilding ? <LoaderCircle size={15} className="animate-spin" /> : <Layers3 size={15} />}Build all pools</button>
              <button className="preset-chip" onClick={exportLabelFile}><Download size={14} />Export labels</button>
            </div>
          </div>
          <p className="mb-4 rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-3 text-sm text-indigo-950">Mark Relevant only if the paper's main topic directly addresses the query's idea. If it only mentions it in passing, or you are unsure, mark Not relevant. Be strict and consistent.</p>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3 text-sm">
            <strong>Benchmark ready: {readyQueryCount} of 12 queries have ≥ 1 relevant label</strong>
            <span className="text-slate-500">Queries with zero relevant labels are excluded from averages.</span>
          </div>
          {labelError && <div role="alert" className="error-banner"><CircleAlert size={18} /><span>{labelError}</span></div>}
          {labelNotice && <div role="status" className="my-3 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">{labelNotice}</div>}
          {exportMessage && <div role="status" className="my-3 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">{exportMessage}</div>}
          <div className="mb-5 rounded-xl border border-slate-200 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold">Query progress</h3><button className="expand-button" onClick={goToNextUnlabeledQuery} disabled={!queries.some((item) => Number(item.pool_count || 0) > Number(item.labeled_count || 0))}>Next unlabeled query <SkipForward size={14} /></button></div>
            <div className="grid max-h-56 gap-2 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3">
              {queries.map((item) => <button key={item.id} className={`rounded-lg border px-3 py-2 text-left text-xs ${item.id === activeQueryId ? 'border-indigo-300 bg-indigo-50' : 'border-slate-200 bg-white'}`} onClick={() => { setLastLabelAction(null); setSkippedPaperIds([]); setActiveQueryId(item.id); }}><span className="block truncate font-medium">{item.text}</span><span className="mt-1 block text-slate-500">{item.labeled_count || 0} / {item.pool_count || 0} labeled · {item.relevant_count || 0} relevant</span></button>)}
            </div>
          </div>
          <div className="query-editor"><label className="sr-only" htmlFor="benchmark-query">Benchmark query</label><select className="topk-select min-w-64" value={activeQueryId || ''} onChange={(event) => { const id = Number(event.target.value); setLastLabelAction(null); setSkippedPaperIds([]); setActiveQueryId(id || null); const selected = queries.find((item) => item.id === id); setQueryText(selected?.text || ''); }}><option value="">Choose a benchmark query</option>{queries.map((item) => <option key={item.id} value={item.id}>{item.text} · {item.labeled_count || 0}/{item.pool_count || 0} labeled</option>)}</select><input id="benchmark-query" className="query-edit-input" value={queryText} onChange={(event) => setQueryText(event.target.value)} placeholder="Add or edit a benchmark query" /><button className="preset-chip" onClick={() => { setActiveQueryId(null); setQueryText(''); }}><Plus size={14} />New</button><button className="search-button" onClick={saveQuery} disabled={!queryText.trim()}>Save query</button></div>
          {activeQueryId && <>
            <div className="label-progress-row"><div><strong>{labelProgress.labeled} / {labelProgress.total} labeled</strong><span> · {queryText} · {pool.filter((paper) => paper.relevant).length} relevant</span></div><span className="count-pill">One pooled list · no method labels</span></div>
            <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-600"><span className="rounded-full bg-slate-100 px-3 py-1.5"><kbd>R</kbd> Relevant</span><span className="rounded-full bg-slate-100 px-3 py-1.5"><kbd>N</kbd> Not relevant</span><span className="rounded-full bg-slate-100 px-3 py-1.5"><kbd>S</kbd> Skip</span><button className="rounded-full bg-slate-100 px-3 py-1.5" onClick={undoLastAction} disabled={!lastLabelAction || labelSaving}><Undo2 size={13} className="mr-1 inline" /><kbd>Z</kbd> Undo last</button><span>Labels auto-advance to the next unlabeled paper.</span></div>
            {labelLoading ? <div className="space-y-3">{[1, 2, 3].map((item) => <div className="skeleton-card" key={item}><div className="skeleton-line w-3/4" /><div className="skeleton-line w-full" /></div>)}</div> : pool.length ? activePaperIndex < 0 ? <div className="empty-results">{skippedPaperIds.length ? <>No remaining unskipped papers. {skippedPaperIds.length} skipped papers remain unlabeled. <button className="expand-button" onClick={() => { setSkippedPaperIds([]); setActivePaperIndex(nextUnlabeledIndex(pool, -1)); }}>Review skipped papers</button></> : <>All papers in this query's pool have been labeled. Choose another query or refresh the pool.</>}</div> : <div className="space-y-3">{pool.map((paper, index) => <LabelCard key={paper.arxiv_id} paper={paper} index={index} active={index === activePaperIndex} onLabel={saveJudgment} saving={labelSaving} />)}</div> : <div className="empty-results">No papers in this query’s pool yet. Use Refresh this pool or Build all pools to prepare candidates.</div>}
          </>}
        </section>}

        {tab === 'benchmark' && <section className="content-panel"><div className="mb-5"><p className="eyebrow"><FlaskConical size={13} /> HUMAN-LABELED EVALUATION</p><h2 className="mt-2 text-2xl font-semibold">Retrieval quality, measured</h2><p className="mt-1 text-sm text-slate-500">Recall and ranking quality are computed from your saved judgments.</p></div>
          {benchmarkError && <div role="alert" className="error-banner"><CircleAlert size={18} />{benchmarkError}</div>}
          {benchmark?.pool_coverage?.status === 'out_of_date' && <div role="alert" className="my-4 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"><CircleAlert size={18} className="mt-0.5 shrink-0" /><div><strong>Label pool needs refresh</strong><p>{benchmark.pool_coverage.warning} ({benchmark.pool_coverage.out_of_pool_count} results)</p></div></div>}
          {benchmarkLoading ? <div className="skeleton-card"><div className="skeleton-line w-2/5" /><div className="skeleton-line w-full" /><div className="skeleton-line w-full" /></div> : benchmark?.status !== 'ok' ? <div className="empty-results"><FlaskConical size={22} /><span>{benchmark?.message || 'No benchmark results yet.'}</span><button className="expand-button" onClick={() => setTab('label')}>Go label papers</button></div> : <>
            <div className="benchmark-headline"><Sparkles size={19} /><strong>{typeof headlineLift === 'number' ? `Hybrid: ${headlineLift >= 0 ? '+' : ''}${headlineLift.toFixed(1)}% Recall@10 over BM25` : 'Human-labeled retrieval benchmark'}</strong><span>Based on {benchmark.num_queries} queries and {benchmark.num_judgments} human judgments{ndcgComparison?.confidence_interval_95 ? ` · NDCG@10 Δ 95% bootstrap CI [${ndcgComparison.confidence_interval_95[0].toFixed(4)}, ${ndcgComparison.confidence_interval_95[1].toFixed(4)}]` : ''}</span></div>
            {ndcgComparison && <p className="mb-4 text-xs text-slate-600">NDCG@10 wins/ties/losses: Hybrid {ndcgComparison.hybrid_wins}, ties {ndcgComparison.ties}, BM25 {ndcgComparison.bm25_wins}. Mean Hybrid − BM25 difference: {ndcgComparison.mean_difference?.toFixed(4) ?? '—'}.</p>}
            <div className="benchmark-caveat">Labels come from one labeler on pooled results. Unlabeled papers are treated as not relevant.</div>
            <div className="overflow-x-auto"><table className="benchmark-table"><thead><tr><th>Method / Embedder</th><th>Recall@5</th><th>Recall@10</th><th>NDCG@5</th><th>NDCG@10</th><th>Query latency</th><th>Build time</th><th>Index size</th></tr></thead><tbody>{Object.entries(benchmark.methods).map(([name, values]) => <tr key={name}><td>{name}</td><td><MetricBar label="" value={values['recall@5']} /></td><td><MetricBar label="" value={values['recall@10']} /></td><td><MetricBar label="" value={values['ndcg@5']} /></td><td><MetricBar label="" value={values['ndcg@10']} /></td><td>{Number(values.latency_ms || 0).toFixed(1)} ms</td><td>{values.index_build_seconds ? `${Number(values.index_build_seconds).toFixed(1)} s` : '—'}</td><td>{values.index_size_bytes ? `${(values.index_size_bytes / (1024 * 1024)).toFixed(1)} MiB` : '—'}</td></tr>)}</tbody></table></div>
            {benchmark.per_query?.length > 0 && <div className="mt-8"><h3 className="mb-3 text-lg font-semibold">Per-query results</h3><div className="overflow-x-auto"><table className="benchmark-table"><thead><tr><th>Query</th><th>BM25 R@10</th><th>Hybrid R@10</th><th>BM25 NDCG@10</th><th>Hybrid NDCG@10</th><th>NDCG@10 winner</th></tr></thead><tbody>{benchmark.per_query.map((item) => { const hybridName = Object.keys(item.methods).find((name) => name.startsWith('hybrid:')) || 'hybrid'; const hybridMethod = item.methods[hybridName]; const bm25 = item.methods.bm25; const ndcgDiff = ((hybridMethod?.['ndcg@10'] || 0) - (bm25?.['ndcg@10'] || 0)) * 100; const winner = ndcgDiff > 0 ? `Hybrid +${ndcgDiff.toFixed(1)} pt` : ndcgDiff < 0 ? `BM25 +${Math.abs(ndcgDiff).toFixed(1)} pt` : 'Tie'; return <tr key={item.query}><td>{item.query}</td><td>{(bm25?.['recall@10'] * 100 || 0).toFixed(1)}%</td><td>{(hybridMethod?.['recall@10'] * 100 || 0).toFixed(1)}%</td><td>{(bm25?.['ndcg@10'] * 100 || 0).toFixed(1)}%</td><td>{(hybridMethod?.['ndcg@10'] * 100 || 0).toFixed(1)}%</td><td className={ndcgDiff > 0 ? 'text-emerald-700' : ndcgDiff < 0 ? 'text-sky-700' : ''}>{winner}</td></tr>; })}</tbody></table></div></div>}
          </>}
        </section>}
        <footer className="mt-14 border-t border-slate-200 pt-5 text-center text-xs text-slate-400">Discover connections across machine learning research.</footer>
      </div>
    </main>
  );
}
