import React from "react";
import axios from "axios";

/*const API_URL = "http://127.0.0.1:8000";*/

export default function SearchBar({ query, onQueryChange, onSearch }) {

  const handleSearch = async (e) => {
    e.preventDefault();
    /*const response = await axios.post(`${API_URL}/search`, {
      query,
    });*/
    onSearch();
  };

  return (
    <form className="search-bar"  onSubmit={handleSearch}>
      <input
        type="text"
        placeholder="Enter manuscript address to add to processing queue"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
      />
    </form>
  );
}
