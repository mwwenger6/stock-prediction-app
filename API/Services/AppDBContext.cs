using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class AppDBContext(DbContextOptions<AppDBContext> options) : DbContext(options)
    {

        public DbSet<User> Users { get; set; }
        public DbSet<Stock> Stocks { get; set; }
        public DbSet<UserStock> UserStocks { get; set; }
        public DbSet<News> News { get; set; }
        public DbSet<UserType> UserTypes { get; set; }
        public DbSet<EODStockPrice> EODStockPrices { get; set; }
        public DbSet<FMStockPrice> FMStockPrices { get; set; }
        public DbSet<QuickStock> QuickStocks { get; set; }
        public DbSet<ErrorLog> ErrorLogs { get; set; }
        public DbSet<UserWatchlistStocks> UserWatchlistStocks { get; set; }
        public DbSet<StockPrediction> StockPredictions { get; set; }
        public DbSet<MarketHolidays> MarketHolidays { get; set; }
    }
}
