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

        public IQueryable<User> GetUsers()
        {
            return dbContext.Users;
        }

        public IQueryable<Stock> GetStocks()
        {
            return dbContext.Stocks;
        }

        public Stock GetStock(string ticker)
        {
            return dbContext.Stocks
                .Where(s => s.Ticker == ticker).Single();
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
