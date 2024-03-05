using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.Extensions.Primitives;
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
        public IQueryable<ErrorLog> GetErrorLogs() => dbContext.ErrorLogs;
        public IQueryable<UserType> GetUserTypes() => dbContext.UserTypes;

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

        public List<UserWatchlistStocks> GetUserWatchlistStocks(int userId)
        {
            return dbContext.UserWatchlistStocks
                .Where(s => s.UserId == userId).ToList();
        }

        public IQueryable<StockPrice> GetStockPrices(string ticker)
        {
            //var dateThreshold = DateTime.UtcNow.AddDays(-daysInterval);
            return dbContext.StockPrices
                .Where(sp => sp.Ticker == ticker) //&& sp.Time >= dateThreshold)
                .OrderByDescending(sp => sp.Time);
        }
        public User GetUser(string email)
        {
            return dbContext.Users
                .Where(u => u.Email == email).Single();
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

        public void AddStockPrices(List<StockPrice> stockPrices)
        {
            using var tempContext = GetNewDBContext();
            foreach (StockPrice stockPrice in stockPrices)
            {
                var existingStockPrice = tempContext.StockPrices
                    .FirstOrDefault(sp => sp.Ticker == stockPrice.Ticker 
                                          && sp.Time == stockPrice.Time);
                if (existingStockPrice == null)
                {
                    tempContext.StockPrices.Add(stockPrice);
                }
            }
            tempContext.SaveChanges();
        }
        public void DeleteStockPrices (string ticker)
        {
            using var tempContext = GetNewDBContext();
            tempContext.StockPrices.Where(s => s.Ticker == ticker).ExecuteDelete();
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


        #endregion
    }
}
