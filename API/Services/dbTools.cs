using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.Extensions.Primitives;
using Pomelo.EntityFrameworkCore.MySql;
using Stock_Prediction_API.Entities;
using BCrypt.Net;

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

        public IQueryable<EODStockPrice> GetStockPrices() => dbContext.EODStockPrices;
        public IQueryable<ErrorLog> GetErrorLogs() => dbContext.ErrorLogs;
        public IQueryable<UserType> GetUserTypes() => dbContext.UserTypes;
        public IQueryable<MarketHolidays> GetMarketHolidays() => dbContext.MarketHolidays;
        public IQueryable<StockPrediction> GetStockPredictions() => dbContext.StockPredictions;

        public Stock GetStock(string ticker)
        {
            return dbContext.Stocks
                .Where(s => s.Ticker == ticker).Single();
        }

        public List<UserWatchlistStocks> GetUserWatchlistStocks(int userId)
        {
            return dbContext.UserWatchlistStocks
                .Where(s => s.UserId == userId).ToList();
        }

        public IQueryable<StockPrice> GetStockPrices(string ticker, bool getHistoricalData)
        {
            if(getHistoricalData)
                return dbContext.EODStockPrices.Where(sp => sp.Ticker == ticker).OrderByDescending(sp => sp.Time);
            else
                return dbContext.FMStockPrices.Where(sp => sp.Ticker == ticker).OrderByDescending(sp => sp.Time);
        }
        public User GetUser(string email)
        {
            return dbContext.Users
                .Where(u => u.Email == email).Single();
        }

        public IQueryable<float> GetStockPredictions(string ticker)
        {
            return dbContext.StockPredictions
                .Where(spred => spred.Ticker == ticker)
                .OrderBy(spred => spred.PredictionOrder)
                .Select(spred => spred.PredictedPrice);
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
                        .SetProperty(t => t.CurrentPrice, stock.CurrentPrice)
                    );
                return;
            }
            tempContext.Stocks.Add(stock);
            tempContext.SaveChanges();
        }
        public void UpdateStockPrice(Stock stock)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.Stocks.Any(s => s.Ticker == stock.Ticker))
            {
                tempContext.Stocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteUpdate(i => i
                        .SetProperty(t => t.CurrentPrice, stock.CurrentPrice)
                        .SetProperty(t => t.UpdatedAt, stock.UpdatedAt)
                        .SetProperty(t => t.DailyChange, stock.DailyChange)
                    );
                return;
            }
            else
                throw new Exception("Stock does not exist");
        }

        public void AddStocks(List<Stock> stocks)
        {
            using var tempContext = GetNewDBContext();
            tempContext.AddRange(stocks);
            tempContext.SaveChanges();
        }

        public void RemoveStock(string ticker)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.Stocks.Any(s => s.Ticker == ticker))
                tempContext.Stocks
                    .Where(s => s.Ticker == ticker)
                    .ExecuteDelete();
        }
        public void AddFMStockPrices(List<StockPrice> stockPrices)
        {
            using var tempContext = GetNewDBContext();
            foreach (StockPrice stockPrice in stockPrices)
            {
                //var existingStockPrice = tempContext.FMStockPrices
                //    .FirstOrDefault(sp => sp.Ticker == stockPrice.Ticker && sp.Time == stockPrice.Time);

                tempContext.FMStockPrices.Add(new()
                {
                    Price = stockPrice.Price,
                    Time = stockPrice.Time,
                    Ticker = stockPrice.Ticker
                });

            }
            tempContext.SaveChanges();
        }
        public void AddEODStockPrices(List<StockPrice> stockPrices)
        {
            using var tempContext = GetNewDBContext();
            foreach (StockPrice stockPrice in stockPrices)
            {
                var existingStockPrice = tempContext.EODStockPrices
                    .FirstOrDefault(sp => sp.Ticker == stockPrice.Ticker && sp.Time == stockPrice.Time);
                if (existingStockPrice == null)
                {
                    tempContext.EODStockPrices.Add(new()
                    {
                        Price = stockPrice.Price,
                        Time = stockPrice.Time,
                        Ticker = stockPrice.Ticker
                    });
                }
            }
            tempContext.SaveChanges();
        }
        public void DeleteStockPrices (string ticker)
        {
            using var tempContext = GetNewDBContext();
            tempContext.EODStockPrices.Where(s => s.Ticker == ticker).ExecuteDelete();
        }
        public void UpdateUserPrivileges(string email, int newUserTypeId)
        {
            using var tempContext = GetNewDBContext();
            tempContext.Users.Where(u => u.Email == email)
                    .ExecuteUpdate(i => i.SetProperty(u => u.TypeId, newUserTypeId));
        }
        public void AddUser(User user)
        {
            using var tempContext = GetNewDBContext();
            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(user.Password); // Hash password before saving
            tempContext.Users.Add(user);
            tempContext.SaveChanges();
        }
        public void AddUserWatchlistStock(UserWatchlistStocks stock)
        {
            using var tempContext = GetNewDBContext();
            tempContext.UserWatchlistStocks.Add(stock);
            tempContext.SaveChanges();
        }
        public void RemoveUserWatchlistStock(int userId, string ticker)
        {
            using var tempContext = GetNewDBContext();
            tempContext.UserWatchlistStocks.Where(s => s.UserId == userId && s.Ticker == ticker).ExecuteDelete();
        }
        public void LogError(ErrorLog error)
        {
            using var tempContext = GetNewDBContext();
            tempContext.ErrorLogs.Add(error);
            tempContext.SaveChanges();
        }

        public void AddPredictions(List<StockPrediction> predictions)
        {
            using var tempContext = GetNewDBContext();
            //foreach (StockPrediction prediction in predictions)
            //{
            //    tempContext.StockPredictions.Add(prediction);
            //}
            tempContext.AddRange(predictions);
            tempContext.SaveChanges();
        }

        public void ClearStockPredictions()
        {
            using var tempContext = GetNewDBContext();
            List<string> tickers = GetStocks().Select(s => s.Ticker).ToList();
            foreach (string ticker in tickers)
            {
                tempContext.StockPredictions.Where(spred => spred.Ticker == ticker).ExecuteDelete();
            }
            tempContext.SaveChanges(); 
        }

        public void AddMarketHolidays(List<MarketHolidays> days)
        {
            using var tempContext = GetNewDBContext();
            tempContext.MarketHolidays.AddRange(days);
            tempContext.SaveChanges();
        }

        #endregion
    }
}
