// frontend/src/components/LoginModal.jsx
import React, { useState } from "react";
import { login } from "../api";

export default function LoginModal({ onClose, onLoginSuccess }) {
  const [username,setUsername]=useState("");
  const [password,setPassword]=useState("");
  const [msg,setMsg]=useState("");

  const submit = async () => {
    try {
      const res = await login(username,password);
      onLoginSuccess(res.data.access_token);
      setMsg("Login correcto");
      onClose();
    } catch (e) {
      setMsg("Error en credenciales");
    }
  };

  return (
    <div className="modal">
      <div className="modal-content">
        <h3>Login</h3>
        <input placeholder="usuario" value={username} onChange={e=>setUsername(e.target.value)} />
        <input type="password" placeholder="contraseña" value={password} onChange={e=>setPassword(e.target.value)} />
        <div style={{marginTop:8}}>
          <button onClick={submit}>Entrar</button>
          <button onClick={onClose} style={{marginLeft:8}}>Cancelar</button>
        </div>
        <p>{msg}</p>
      </div>
    </div>
  );
}
