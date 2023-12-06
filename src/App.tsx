import React from 'react';
import './App.css';
import HomeView from "./Views/HomeView";
import AppNavbar from "./Components/AppNavbar";
import NewsView from './Views/NewsView';
import StockGraphView from './Views/StockGraphView';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Ticker from './Components/Ticker';

function App() {

  return (
    <div className="App">
      <BrowserRouter>
        <Ticker/>
        <AppNavbar/>
        <Routes>
          <Route index element = { <HomeView/>} />
          <Route path="News" element = { <NewsView/> } />
          <Route path="Stock/:symbol" element = { <StockGraphView/> } />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
