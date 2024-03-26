import React, {useEffect, useState} from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css';
import HomeView from "./Views/HomeView";
import AppNavbar from "./Components/AppNavbar";
import VerificationView from "./Views/EmailVerifiedView";
import NewsView from './Views/NewsView';
import StockGraphView from './Views/StockGraphView';
import User from "./Interfaces/User"
import Stock from "./Interfaces/Stock";
import TickerScroller from "./Components/TickerScroller";
import AccountView from './Views/AccountView';
import AdminView from './Views/AdminView';
import GetFeaturedStocks from "./Services/GetFeaturedStocks";
import GetWatchListStocks from "./Services/GetWatchListStocks";
import GetPersonalStocks from "./Services/GetPersonalStocks";
import config from "./config";

const initFeaturedStocks: Stock[] = [];
const initWatchListStocks: Stock[] = [];
const initPersonalStocks: Stock[] = [];

function App() {

  const getFeaturedStocks = GetFeaturedStocks;
  const getWatchListStocks = GetWatchListStocks;
  const getPersonalStocks = GetPersonalStocks;

  const [user, setUser] = useState<User | null>(null);
  const [featuredStocks, setFeaturedStocks] = useState(initFeaturedStocks)
  const [watchListStocks, setWatchListStocks] = useState(initWatchListStocks)
  const [homeViewStocks, setHomeViewStocks] = useState(initWatchListStocks)
  const [personalStocks, setPersonalStocks] = useState(initPersonalStocks)

  //Fetch featured stocks price data on initial load
  useEffect(() => {
    const fetchFeatured = async () => {
      try {
        const stocks: Stock[] | null = await getFeaturedStocks();
        if (stocks !== null) {
          setFeaturedStocks(stocks);
          setHomeViewStocks(stocks);
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
        setHomeViewStocks(featuredStocks)
        setWatchListStocks(initWatchListStocks)
        setPersonalStocks(initPersonalStocks)
        return;
      }
      const stocks: Stock[] | null = await getWatchListStocks(user.id);
      const personalStockList: Stock[] | null = await getPersonalStocks(user.id);
      if (stocks !== null) {
        setWatchListStocks(stocks);
        setHomeViewStocks(stocks);
      }
      if (personalStockList !== null) {
        setPersonalStocks(personalStockList)
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
        <Routes>
          <Route index element={<HomeView user={user} homeviewStocks={homeViewStocks} personalStocks={personalStocks}/>} />
          <Route path="Verification/:code" element= {<VerificationView/> } />
          <Route path="News" element = { <NewsView /> } />
          <Route path="Settings/Account" element={<AccountView user={user} />} />
          <Route path="Settings/Admin" element={<AdminView />} />
          <Route path="Stock/:symbol" element = { <StockGraphView user={user} featuredStocks={featuredStocks} watchlistStocks={watchListStocks} reloadWatchlist = { () => fetchStocks() }/> } />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
