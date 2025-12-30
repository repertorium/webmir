import React from "react";
import logo from "../assets/logo-rep-black.d0671a34.svg";

export default function Header({ onLoginClick }) {
  return (
    <header className="header">
      <div className="header-left">
        <img src={logo} alt="Logo" className="logo-img" />
        <button className="login-link" onClick={onLoginClick}>
          Login
        </button>
      </div>
    </header>
  );
}
