import React, { useEffect, useState } from "react";
import Header from "./components/Header";
import SearchBar from "./components/SearchBar";
import Results from "./components/Results";
import AuthModal from "./components/AuthModal";
import ManuscriptsList from "./components/ManuscriptsList";
import { getManuscripts, getChants, enqueue } from "./api";
import "./App.css";


export default function App() {
  const [data, setData] = useState({processed:[], processing:{running:null, queue:[]}});
  const [selected, setSelected] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [showAuth, setShowAuth] = useState(false);
  const [token, setToken] = useState(null);


  const refresh = async () => {
    try {
      const res = await getManuscripts();
      setData(res.data);      
    }
    catch (e) {
      console.error(e);
    }
  };

  useEffect(()=>{ refresh(); const id = setInterval(refresh, 5000); return ()=>clearInterval(id); }, []);

  const onSelect = async (ms_id) => {
    setSelected(ms_id);
    try {
      const res = await getChants(ms_id);
      setResults(res.data);
    } 
    catch (e) {
      console.error(e);
      setResults([]);
    }
  };

  const handleEnqueue = async () => {
    if (!token) { alert("You need to log in"); return; }
    try {
      await enqueue(query, token);
      setQuery("");
      refresh();
    } catch (e) {
      alert("Error al encolar: " + (e.response?.data?.detail || e.message));
    }
  };

  return (
    <div>
      <Header onLoginClick={() => setShowAuth(true)} />
      <main className="main">
        <SearchBar query={query} onQueryChange={setQuery} onSearch={handleEnqueue} />
        <ManuscriptsList processed={data.processed} processing={data.processing} onSelect={onSelect} />
        <Results results={results} />
      </main>
      
      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onLoginSuccess={(t) => {
            setToken(t);
            setShowAuth(false);
          }}
        />
      )}
    </div>
  );
}
