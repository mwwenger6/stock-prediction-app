using Stock_Prediction_API.Controllers;
using Stock_Prediction_API.Entities;

namespace Stock_Prediction_API.Services
{
    public class MyBackgroundService : BackgroundService
    {
        private readonly ILogger<MyBackgroundService> _logger;
        private readonly IServiceProvider _serviceProvider;

        public MyBackgroundService(ILogger<MyBackgroundService> logger, IServiceProvider serviceProvider)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                _logger.LogInformation("Background service running");

                if (IsStockMarketHours(DateTime.Now))
                {

                    using (var scope = _serviceProvider.CreateScope())
                    {
                        var homeController = scope.ServiceProvider.GetRequiredService<HomeController>();
                        await homeController.AddRecentStockPrices();
                    }

                }
                else
                {
                    _logger.LogInformation("No price updates as stock market is closed.");
                }

                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken); // Delay for 5 minutes
            }
            bool IsStockMarketHours(DateTime dateTime)
            {
                // Check if the time is between 9:30 AM and 4:00 PM (Eastern Time) on weekdays
                return dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday
                    && dateTime.Hour >= 9 && dateTime.Hour <24 && (dateTime.Hour != 9 || dateTime.Minute >= 30);
            }
        }
    }
}