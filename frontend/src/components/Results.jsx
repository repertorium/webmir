import React from "react";

export default function Results({ results }) {
  if (!results || results.length === 0) {
    return (
      <div className="results">
        <p>No hay resultados para mostrar.</p>
      </div>
    );
  }
  return (
    <div className="results">
      <h2>List of chants</h2>
      <div className="results-scroll">
      <table className="results-table">
        <thead>
          <tr>
            <th>Cantus ID</th>
            <th>Text</th>
            <th>Prob</th>
            <th>Page</th>
            <th>Link</th>
          </tr>
        </thead>
        <tbody>
          {results.map((item, index) => {
            let rowClass = "";

            if (item.cantusid === "unknown") {
              rowClass = "row-unknown";
            } else if (item.prob < 0.6) {
              rowClass = "row-lowprob";
            }

            return (
              <tr key={index} className={rowClass}>
                <td>
                  <a
                    href={`https://cantusindex.org/id/${item.cantusid}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {item.cantusid}
                  </a>
                </td>
                <td>{item.lyrics}</td>
                <td>{item.prob}</td>
                <td>{item.page}</td>
                <td>
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Image link
                  </a>
                </td>
              </tr>
            );
          })}
          </tbody>
      </table>
      </div>
    </div>
  );
}
