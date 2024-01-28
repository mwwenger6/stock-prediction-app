using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;

namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config) : base(context, config) {}

        //[HttpGet("/Home/GetUsers")]
        //public IActionResult GetUsers()
        //{
        //    try
        //    {
        //        List<User> user = _GetDataTools.GetUsers().ToList();
        //        int id = user.First().Id;
        //    }
        //    catch(Exception ex)
        //    {

        //    }
        //    return View();
        //}


        [HttpGet("/Home/GetRecentStockPrice/{ticker}")]
        public IActionResult GetRecentStockPrice(string ticker)
        {
            try
            {
                StockPrice recentPrice = _GetDataTools.GetRecentStockPrice(ticker);
                if (recentPrice == null)
                {
                    return NotFound("Stock price not found.");
                }

                return Json(recentPrice);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }


        [HttpGet("/Home/GetStockPrices/{ticker}/{interval}")]
        public IActionResult GetStockPrices(string ticker, int interval)
        {
            try
            {
                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker, interval).ToList();
                if (stockPrices == null || !stockPrices.Any())
                {
                    return NotFound("Stock prices not found.");
                }

                return Json(stockPrices);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }

        // [HttpGet("/Home/AddStockPrices/{ticker}/{interval}")]
        // public IActionResult AddStockPrices(string ticker)
        //{
        // }

        [HttpGet("/Home/AddStockPrices")]
        public IActionResult AddStockPrices()
        {
            try
            {
                List<string> quickStockTickers = _GetDataTools.GetQuickStocks().Select(qs => qs.Ticker).ToList();
                List<StockPrice> stockPrices = new List<StockPrice>();

                foreach (string ticker in quickStockTickers)
                {
                    var price = GetPriceForTicker(ticker); // Assume this is a method to get the price

                    stockPrices.Add(new StockPrice
                    {
                        Ticker = ticker,
                        Price = (float)price,
                        Time = DateTime.UtcNow // Or the appropriate time
                    });
                }

                _GetDataTools.AddStockPrices(stockPrices);
                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }
        private decimal GetPriceForTicker(string ticker)
        {
            // Implement the logic to determine the price for a given ticker
            throw new NotImplementedException();
        }



        [HttpGet("/Home/AddStockPricesByBatch")]
        public IActionResult AddStockPricesByBatch()
        {
            try
            {
                List<string> quickStockTickers = _GetDataTools.GetQuickStocks()
                                                              .Select(qs => qs.Ticker)
                                                              .ToList();
                List<StockPrice> stockPrices = GetPriceForTickers(quickStockTickers);

                _GetDataTools.AddStockPrices(stockPrices);
                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }

        private List<StockPrice> GetPriceForTickers(List<string> tickers)
        {
            // Implement the logic to make a batch request to Twelvedata
            // and parse the response to create a list of StockPrice objects.
            // This is a placeholder implementation.
            throw new NotImplementedException();
        }



    }
}
