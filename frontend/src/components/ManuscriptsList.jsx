// frontend/src/components/ManuscriptsList.jsx
import React from "react";

export default function ManuscriptsList({ processed, processing, onSelect }) {
  const handleProcessedChange = (e) => {
    const value = e.target.value;
    if (value) onSelect(value);
  };

  const running = processing?.running;
  const queue = processing?.queue || [];

  return (
    <div className="manuscripts-container">
      <section className="manuscripts-box">
        <h3>Processed manuscripts</h3>

        <select defaultValue="" onChange={handleProcessedChange}>
          <option value="" disabled>
            Select a manuscript…
          </option>

          {processed.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </section>

      <section className="manuscripts-box">
        <h3>In queue manuscripts</h3>

        <select>
          {running ? (
            <option>▶ In excution: {running.manuscript_id}</option>
          ) : (
            <option>No manuscripts in execution</option>
          )}

          {queue.length > 0 && (
            <>
              <option disabled>──────────</option>
              {queue.map((t) => (
                <option key={t.id}>In queue: {t.manuscript_id}</option>
              ))}
            </>
          )}
        </select>
      </section>
    </div>
  );
}
