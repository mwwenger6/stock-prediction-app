using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    /// <summary>
    /// Represents the database context for the Stock Prediction API, containing sets of entities managed by the context.
    /// </summary>
    public class AppDBContext(DbContextOptions<AppDBContext> options) : DbContext(options)
    {
        /// <summary>
        /// Gets or sets the set of Users.
        /// </summary>
        public DbSet<User> Users { get; set; }

        /// <summary>
        /// Gets or sets the set of Stocks.
        /// </summary>
        public DbSet<Stock> Stocks { get; set; }

        /// <summary>
        /// Gets or sets the set of UserStocks, representing the association between users and their stocks.
        /// </summary>
        public DbSet<UserStock> UserStocks { get; set; }

        /// <summary>
        /// Gets or sets the set of News articles related to stocks.
        /// </summary>
        public DbSet<News> News { get; set; }

        /// <summary>
        /// Gets or sets the set of UserTypes, categorizing users by their roles or types.
        /// </summary>   
        public DbSet<UserType> UserTypes { get; set; }

        /// <summary>
        /// Gets or sets the set of StockPrices, containing historical and current price data for stocks.
        /// </summary>
        public DbSet<StockPrice> StockPrices { get; set; }

        /// <summary>
        /// Gets or sets the set of QuickStocks, intended for fast-access stock information.
        /// </summary>
        public DbSet<QuickStock> QuickStocks { get; set; }

        /// <summary>
        /// Gets or sets the set of ErrorLogs, used for logging application errors.
        /// </summary>
        public DbSet<ErrorLog> ErrorLogs { get; set; }

        /// <summary>
        /// Gets or sets the set of UserWatchlistStocks, representing stocks that users have added to their watchlists.
        /// </summary>
        public DbSet<UserWatchlistStocks> UserWatchlistStocks { get; set; }

        /// <summary>
        /// Gets or sets the set of StockPredictions, containing predicted future stock prices.
        /// </summary>
        public DbSet<StockPrediction> StockPredictions { get; set; }

        /// <summary>
        /// Gets or sets the set of MarketHolidays, indicating days when the stock market is closed.
        /// </summary>
        public DbSet<MarketHolidays> MarketHolidays { get; set; }

        /// <summary>
        /// Gets or sets the set of SupportedStocks, listing stocks that are tracked and supported by the application.
        /// </summary>
        public DbSet<SupportedStocks> SupportedStocks { get; set; }
    }
}
