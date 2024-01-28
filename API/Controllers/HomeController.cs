using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.ViewModel;

namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config) : base(context, config) {}

        [HttpGet("/Home/GetData")]
        public IActionResult GetData()
        {
            try
            {
                List<User> user = _GetDataTools.GetUsers().ToList();
                List<QuickStock> quickStocks = _GetDataTools.GetQuickStocks().ToList();
                List<Stock> stocks = _GetDataTools.GetStocks().ToList();
                List<StockPrice> prices =  _GetDataTools.GetStockPrices().ToList();
            }
            catch (Exception ex)
            {

            }
            return View();
        }

        //[HttpGet("/Home/AddData")]
        //public IActionResult AddData()
        //{
        //    try
        //    {
        //        Stock stock = new()
        //        {
        //            Ticker = "GOOG",
        //            CreatedAt = DateTime.Now,
        //            Name = "Google"
        //        };
        //        Stock msft = new()
        //        {
        //            Ticker = "MSFT",
        //            CreatedAt = DateTime.Now,
        //            Name = "Microsoft"
        //        };
        //        List<Stock> stocks = new();
        //        stocks.Add(stock);
        //        stocks.Add(msft);
        //        _GetDataTools.AddStocks(stocks);
        //        return Ok("Stock prices added successfully.");
        //    }
        //    catch (Exception ex)
        //    {
        //        // Log the exception
        //        return StatusCode(500, "Internal server error");
        //    }
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
        public async Task<IActionResult> AddStockPricesByBatch()
        {
            try
            {
                string quickStockTickers = string.Join(",", _GetDataTools.GetQuickStocks().Select(qs => qs.Ticker));
                string interval = "5min";
                string stockPrices = await _TwelveDataTools.GetPriceForTickers(quickStockTickers, interval) ?? throw new Exception("Data is Null");
                var vmDict = JsonConvert.DeserializeObject<Dictionary<string, TwelveDataViewModel>>(stockPrices);
                List<StockPrice> stockPriceList = vmDict.Select(kv => new StockPrice
                {
                    Ticker = kv.Value.Symbol,
                    Price = float.Parse(kv.Value.Close),
                    Time = kv.Value.DateTime
                }).ToList();

                _GetDataTools.AddStockPrices(stockPriceList);
                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }

        private List<StockPrice> GetPriceForTickers(string tickers)
        {
            // Implement the logic to make a batch request to Twelvedata
            // and parse the response to create a list of StockPrice objects.
            // This is a placeholder implementation.
            throw new NotImplementedException();
        }



    }
}
