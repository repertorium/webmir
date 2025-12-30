import React, { useState } from "react";
import { login, register } from "../api";

export default function AuthModal({ onClose, onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [message, setMessage] = useState("");

  const handleSubmit = async () => {
    try {
      if (mode === "login") {
        const res = await login(username, password);
        onLoginSuccess(res.data.access_token);
        setMessage("✅ Sesión iniciada");
      } else {
        await register(username, password);
        setMessage("✅ Usuario registrado correctamente");
      }
    } catch {
      setMessage("❌ Error en autenticación");
    }
  };

  return (
    <div className="modal">
      <div className="modal-content">
        <h3>{mode === "login" ? "Iniciar Sesión" : "Registrar Usuario"}</h3>
        <input
          placeholder="User"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button onClick={handleSubmit}>
          {mode === "login" ? "Enter" : "Register (not available)"}
        </button>
        <p
          className="switch"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login"
            ? "¿No tienes cuenta? Regístrate"
            : "¿Ya tienes cuenta? Inicia sesión"}
        </p>
        <p className="msg">{message}</p>
        <button className="close" onClick={onClose}>
          Cerrar
        </button>
      </div>
    </div>
  );
}
