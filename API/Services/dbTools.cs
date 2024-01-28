﻿using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Pomelo.EntityFrameworkCore.MySql;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class dbTools
    {

        private readonly AppDBContext dbContext;
        private readonly DbContextOptions<AppDBContext> dbContextOptions;

        public dbTools(AppDBContext context, DbContextOptions<AppDBContext> options)
        {
            dbContext = context;
            dbContextOptions = options;
        }

        private AppDBContext GetNewDBContext()
        {
            return new AppDBContext(dbContextOptions);
        }

        //Getters
        #region Getters

        public IQueryable<User> GetUsers() => dbContext.Users;

        public IQueryable<Stock> GetStocks() => dbContext.Stocks;

        public IQueryable<QuickStock> GetQuickStocks() => dbContext.QuickStocks;

        public IQueryable<StockPrice> GetStockPrices() => dbContext.StockPrices;


        public Stock GetStock(string ticker)
        {
            return dbContext.Stocks
                .Where(s => s.Ticker == ticker).Single();
        }

        public StockPrice GetRecentStockPrice(string ticker)
        {
            return dbContext.StockPrices
                .Where(sp => sp.Ticker == ticker)
                .OrderByDescending(sp => sp.Time)
                .FirstOrDefault();
        }

        public IQueryable<StockPrice> GetStockPrices(string ticker, int daysInterval)
        {
            var dateThreshold = DateTime.UtcNow.AddDays(-daysInterval);
            return dbContext.StockPrices
                .Where(sp => sp.Ticker == ticker && sp.Time >= dateThreshold)
                .OrderByDescending(sp => sp.Time);
        }


        #endregion

        //Modifiers
        #region Modifiers
        public void AddStock(Stock stock)
        {
            using var tempContext = GetNewDBContext();
            if(tempContext.Stocks.Any(s => s.Ticker == stock.Ticker))
            {
                tempContext.Stocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteUpdate(i => i
                        .SetProperty(t => t.Name, stock.Name)
                        .SetProperty(t => t.OneDayPercentage, stock.OneDayPercentage)
                    );
                return;
            }
            tempContext.Stocks.Add(stock);
            tempContext.SaveChanges();
        }

        public void AddStocks(List<Stock> stocks)
        {
            using var tempContext = GetNewDBContext();
            tempContext.AddRange(stocks);
            tempContext.SaveChanges();
        }

        public void RemoveStock(Stock stock)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.Stocks.Any(s => s.Ticker == stock.Ticker))
                tempContext.Stocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteDelete();
        }

        public void AddStockPrices(List<StockPrice> stockPrices)
        {
            using var tempContext = GetNewDBContext();
            //tempContext.StockPrices.AddRange(stockPrices);
            foreach (StockPrice stockPrice in stockPrices)
            {
                tempContext.StockPrices.Add(stockPrice);
            }
            tempContext.SaveChanges();
        }





        #endregion
    }
}
