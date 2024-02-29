import React, {useEffect, useState} from 'react';
import './App.css';
import HomeView from "./Views/HomeView";
import AppNavbar from "./Components/AppNavbar";
import NewsView from './Views/NewsView';
import LoginView from './Views/LoginView';
import StockGraphView from './Views/StockGraphView';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import User from "./Interfaces/User"
import Stock from "./Interfaces/Stock";
import TickerScroller from "./Components/TickerScroller";
import GetFeaturedStocks from "./Services/GetFeaturedStocks";
import ErrorLogsView from './Views/ErrorLogsView';


//set initial featured stocks list
const initFeaturedStocks: Stock[] = [
  { name: 'Apple', ticker: 'AAPL', price: -1, up: undefined},
  { name: 'Google', ticker: 'GOOGL', price: -1, up: undefined},
  { name: 'Amazon', ticker: 'AMZN', price: -1, up: undefined },
  { name: 'Microsoft', ticker: 'MSFT', price: -1, up: undefined },
  { name: 'Meta Platforms', ticker: 'META', price: -1, up: undefined },
  { name: 'Tesla', ticker: 'TSLA', price: -1, up: undefined },
  { name: 'Netflix', ticker: 'NFLX', price: -1, up: undefined },
  { name: 'Alphabet', ticker: 'GOOG', price: -1, up: undefined },
  { name: 'Visa', ticker: 'V', price: -1, up: undefined },
  { name: 'Procter & Gamble', ticker: 'PG', price: -1, up: undefined },
  { name: 'Cisco Systems', ticker: 'CSCO', price: -1, up: undefined },
  { name: 'JPMorgan Chase', ticker: 'JPM', price: -1, up: undefined },
  { name: 'Coca-Cola', ticker: 'KO', price: -1, up: undefined },
  { name: 'Adobe', ticker: 'ADBE', price: -1, up: undefined },
  { name: 'PayPal', ticker: 'PYPL', price: -1, up: undefined },
  { name: 'Home Depot', ticker: 'HD', price: -1, up: undefined },
];

function App() {

  const getFeaturedStocks = GetFeaturedStocks;

  const [user, setUser] = useState<User | null>(null);
  const [featuredStocks, setFeaturedStocks] = useState(initFeaturedStocks)

  //Fetch featured stocks price data on load
  useEffect(() => {
    const fetchData = async () => {
      try {
        const stocks: Stock[] | null = await getFeaturedStocks();
        if (stocks !== null) {
          setFeaturedStocks(stocks);
        }
      } catch (error) {
        console.error('Error fetching prices:', error);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="App">
      <TickerScroller featuredStocks={featuredStocks}/>
      <BrowserRouter>
        <AppNavbar user={user} setUser={setUser}/>
        <Routes>
          <Route index element={<HomeView user={user} featuredStocks={featuredStocks}/>} />
          <Route path="News" element = { <NewsView /> } />
          <Route path="Login" element = { <LoginView/> } />
          <Route path="Admin/ErrorLogs" element={<ErrorLogsView />} />
          <Route path="Stock/:symbol" element = { <StockGraphView user={user} featuredStocks={featuredStocks} /> } />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
