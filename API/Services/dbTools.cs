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

        public IQueryable<StockPrice> GetStockPrices() => dbContext.StockPrices;
        public IQueryable<ErrorLog> GetErrorLogs() => dbContext.ErrorLogs;
        public IQueryable<UserType> GetUserTypes() => dbContext.UserTypes;
        public IQueryable<MarketHolidays> GetMarketHolidays() => dbContext.MarketHolidays;
        public IQueryable<StockPrediction> GetStockPredictions() => dbContext.StockPredictions;

        #region Stocks

        public List<SupportedStocks> GetSupportedStocks()
        {
            return dbContext.SupportedStocks.ToList();
        }

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

        public IQueryable<StockPrice> GetStockPrices(string ticker, bool getHistoricalData, DateTime? startDate = null)
        {
            IQueryable<StockPrice> prices;
            if(getHistoricalData)
                prices = dbContext.StockPrices.Where(sp => sp.Ticker == ticker && sp.IsClosePrice == true).OrderByDescending(sp => sp.Time);
            else
                prices = dbContext.StockPrices.Where(sp => sp.Ticker == ticker).OrderByDescending(sp => sp.Time);

            if (startDate != null)
                prices = prices.Where(s => s.Time >= startDate);

            return prices;
        }

        public IQueryable<StockPrice> GetStockPrices(string ticker, DateTime date)
        {
            return dbContext.StockPrices
                .Where(sp => sp.Ticker == ticker && sp.Time >= date)
                .OrderByDescending(sp => sp.Time);
        }


        #endregion

        #region User

        public User GetUser(string email)
        {
            return dbContext.Users
                .Where(u => u.Email == email).Single();
        }
        public IQueryable<UserStock> GetUserStocks(int userId)
        {
            return dbContext.UserStocks.Where(u => u.UserId == userId);
        }
        public UserStock GetUserStock(int userId, string ticker)
        {
            return dbContext.UserStocks
                .Single(u => u.UserId == userId && u.Ticker == ticker);
        }
        public UserStock? GetUserStockNullable(int userId, string ticker)
        {
            return dbContext.UserStocks
                .FirstOrDefault(u => u.UserId == userId && u.Ticker == ticker);
        }
        public bool UserWithVerificationCode(string code)
        {
            return dbContext.Users.Any(u => u.VerificationCode == code);
        }
        public User GetUserByVerificationCode(string code)
        {
            return dbContext.Users.Where(u => u.VerificationCode == code).FirstOrDefault();
        }
        #endregion

        #region Predictions
        public IQueryable<float> GetStockPredictions(string ticker)
        {
            return dbContext.StockPredictions
                .Where(spred => spred.Ticker == ticker)
                .OrderBy(spred => spred.PredictionOrder)
                .Select(spred => spred.PredictedPrice);
        }
        #endregion

        #region Discovery

        public IQueryable<VolatileStock> GetVolatileStocks(bool isPositive)
        {
            return dbContext.VolatileStocks
                .Where(s => s.IsPositive == isPositive);
        }
        #endregion

        #endregion

        //Modifiers
        #region Modifiers

        #region Stocks
        public void AddSupportedStocks(List<SupportedStocks> supportedStocks)
        {
            using var tempContext = GetNewDBContext();
            tempContext.SupportedStocks.AddRange(supportedStocks);
            tempContext.SaveChanges();
        }

        public int RemoveSupportedStocks()
        {
            using var tempContext = GetNewDBContext();
            IQueryable<SupportedStocks> stocks = tempContext.SupportedStocks.Where(s => s.Ticker != null);
            int count = stocks.Count();
            stocks.ExecuteDelete();
            return count;
        }

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

        public void AddUserStock(UserStock stock)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.UserStocks.Any(s => s.Ticker == stock.Ticker))
            {
                if (stock.Quantity < 0) return;
                if (stock.Quantity == 0)
                {
                    tempContext.UserStocks
                        .Where(s => s.Ticker == stock.Ticker)
                        .ExecuteDelete();
                    return;
                }
                tempContext.UserStocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteUpdate(i => i
                        .SetProperty(t => t.Quantity, stock.Quantity)
                        .SetProperty(t => t.Price, stock.Price)
                    );
                return;
            }
            if (stock.Quantity <= 0) return;
            tempContext.UserStocks.Add(stock);
            tempContext.SaveChanges();
        }
        public void RemoveUserStock(UserStock stock)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.UserStocks.Any(s => s.Ticker == stock.Ticker))
                tempContext.UserStocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteDelete();
            else
                throw new Exception("User does not have any shares of that stock.");
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
        public void AddStockPrices(List<StockPrice> stockPrices)
        {
            using var tempContext = GetNewDBContext();
            tempContext.StockPrices.AddRange(stockPrices);
            tempContext.SaveChanges();
        }
        public void DeleteStockPrices (string ticker)
        {
            using var tempContext = GetNewDBContext();
            tempContext.StockPrices.Where(s => s.Ticker == ticker).ExecuteDelete();
        }
        #endregion

        #region User
        public void AddUser(User user)
        {
            using var tempContext = GetNewDBContext();
                // Hash password with BCrypt before saving
                user.Password = BCrypt.Net.BCrypt.HashPassword(user.Password);
                if (tempContext.Users.Any(u => u.Id == user.Id))
            {
                tempContext.Users.Where(u => u.Id == user.Id)
                    .ExecuteUpdate(u => u
                        .SetProperty(u => u.IsVerified, user.IsVerified)
                        .SetProperty(u => u.VerificationCode, user.VerificationCode)
                    );
            }
            else
            {
                tempContext.Users.Add(user);
                tempContext.SaveChanges();
            }
        }
        public void DeleteUser(string email)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.Users.Any(s => s.Email == email))
            {
                tempContext.Users
                    .Where(s => s.Email == email)
                    .ExecuteDelete();
                return;
            }
        }
        public void UpdateUserPrivileges(string email, int newUserTypeId)
        {
            using var tempContext = GetNewDBContext();
            tempContext.Users.Where(u => u.Email == email)
                    .ExecuteUpdate(i => i.SetProperty(u => u.TypeId, newUserTypeId));
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
        #endregion

        #region Predictions


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

        #endregion

        #region Other

        public void AddMarketHolidays(List<MarketHolidays> days)
        {
            using var tempContext = GetNewDBContext();
            tempContext.MarketHolidays.AddRange(days);
            tempContext.SaveChanges();
        }
        public void LogError(ErrorLog error)
        {
            using var tempContext = GetNewDBContext();
            tempContext.ErrorLogs.Add(error);
            tempContext.SaveChanges();
        }
        public void DeleteErrorLog(ErrorLog log)
        {
            using var tempContext = GetNewDBContext();
            tempContext.ErrorLogs.Remove(log);
            tempContext.SaveChanges();
        }
        #endregion

        #endregion
    }
}
