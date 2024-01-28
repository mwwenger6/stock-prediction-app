using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class dbTools
    {

        private readonly AppDBContext dbContext;

        public dbTools(AppDBContext context)
        {
            dbContext = context;
        }

        private AppDBContext GetNewDBContext()
        {
            return new(new DbContextOptionsBuilder<AppDBContext>()
                .UseSqlServer(dbContext.Database.GetConnectionString()).Options);
        }

        //Getters
        #region Getters

        public IQueryable<User> GetUsers() => dbContext.Users;

        public IQueryable<Stock> GetStocks() => dbContext.Stocks;

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

        public void RemoveStock(Stock stock)
        {
            using var tempContext = GetNewDBContext();
            if (tempContext.Stocks.Any(s => s.Ticker == stock.Ticker))
                tempContext.Stocks
                    .Where(s => s.Ticker == stock.Ticker)
                    .ExecuteDelete();
        }


        #endregion
    }
}
