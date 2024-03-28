import React, {useEffect, useState} from 'react';
import { BrowserRouter, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import './App.css';
import HomeView from "./Views/HomeView";
import AppNavbar from "./Components/AppNavbar";
import VerificationView from "./Views/EmailVerifiedView";
import DiscoveryView from './Views/DiscoveryView';
import StockGraphView from './Views/StockGraphView';
import User from "./Interfaces/User"
import Stock from "./Interfaces/Stock";
import TickerScroller from "./Components/TickerScroller";
import AccountView from './Views/AccountView';
import AdminView from './Views/AdminView';
import GetFeaturedStocks from "./Services/GetFeaturedStocks";
import GetWatchListStocks from "./Services/GetWatchListStocks";
import NotFoundPage from './Views/NotFoundPage';
import config from "./config";

const initFeaturedStocks: Stock[] = [];
const initWatchListStocks: Stock[] = [];
const initPersonalStocks: Stock[] = [];

function App() {

  const getFeaturedStocks = GetFeaturedStocks;
  const getWatchListStocks = GetWatchListStocks;

  const [user, setUser] = useState<User | null>(null);
  const [featuredStocks, setFeaturedStocks] = useState(initFeaturedStocks)
  const [watchListStocks, setWatchListStocks] = useState(initWatchListStocks)
  const isLoggedIn = user != null;

  //Fetch featured stocks price data on initial load
  useEffect(() => {
    const fetchFeatured = async () => {
      try {
        const stocks: Stock[] | null = await getFeaturedStocks();
        if (stocks !== null) {
          setFeaturedStocks(stocks);
        }
      } catch (error) {
        console.error('Error fetching prices:', error);
      }
    };
    fetchFeatured();
  }, []);

  //Fetch user watchlist stocks on user change
  useEffect(() => {
    fetchStocks();
  }, [user]);

  const fetchStocks = async () => {
    try {
      if(user === null) {
        setWatchListStocks(initWatchListStocks)
        return;
      }
      const stocks: Stock[] | null = await getWatchListStocks(user.id);
      if (stocks !== null) {
        setWatchListStocks(stocks);
      }
    } catch (error) {
      console.error('Error fetching prices:', error);
    }
  };

  return (
    <div className="App">
      <TickerScroller featuredStocks={featuredStocks}/>
      <BrowserRouter>
        <AppNavbar user={ user } setUser={setUser}/>
        {isLoggedIn ? (
            <Routes>
              <Route path="Verification/:code" element= {<VerificationView/> } />
              <Route path="Stock/:symbol" element = { <StockGraphView user={user} featuredStocks={featuredStocks} watchlistStocks={watchListStocks} reloadWatchlist = { () => fetchStocks() }/> } />
              <Route path="/" element = { <DiscoveryView featuredStocks={featuredStocks} /> } />
              <Route path="Home" index element={<HomeView user={user} watchlistStocks={watchListStocks}/>} />
              <Route path="Settings/Account" element={<AccountView user={user} />} />
              <Route path="Settings/Admin" element={<AdminView />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
        ) :
          <>
            <Routes>
              <Route path="Verification/:code" element= {<VerificationView/> } />
              <Route path="Stock/:symbol" element = { <StockGraphView user={user} featuredStocks={featuredStocks} watchlistStocks={watchListStocks} reloadWatchlist = { () => fetchStocks() }/> } />
              <Route path="/" index element = { <DiscoveryView featuredStocks={featuredStocks} /> } />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </>
        }
      </BrowserRouter>
    </div>
  );
}

export default App;
