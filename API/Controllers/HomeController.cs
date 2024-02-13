using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MySqlConnector;
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

        [HttpGet("/Home/GetUser/{email}")]
        public IActionResult GetUser(string email)
        {
            try
            {
                User user = _GetDataTools.GetUser(email);
                return Json(user);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }
        [HttpGet("/Home/AuthenticateUser/{email}/{password}")]
        public IActionResult AuthenticateUser(string email, string password)
        {
            try
            {
                User user = _GetDataTools.GetUser(email);
                if (user.Password != password)
                    throw new InvalidDataException("Could not authenticate");
                return Json(user);
            }
            catch (InvalidDataException ex)
            { 
                //Log the exception
                return StatusCode(401, ex.Message);
            }
            catch (Exception ex)
            {
                //Log the exception
                return StatusCode(500, "Internal server error");
            }
        }

        // [HttpGet("/Home/AddStockPrices/{ticker}/{interval}")]
        // public IActionResult AddStockPrices(string ticker)
        //{
        // }

        //[HttpGet("/Home/AddStockPrices")]
        //public IActionResult AddStockPrices()
        //{
        //    try
        //    {
        //        List<string> quickStockTickers = _GetDataTools.GetQuickStocks().Select(qs => qs.Ticker).ToList();
        //        List<StockPrice> stockPrices = new List<StockPrice>();

        //        foreach (string ticker in quickStockTickers)
        //        {
        //            var price = GetPriceForTicker(ticker); // Assume this is a method to get the price

        //            stockPrices.Add(new StockPrice
        //            {
        //                Ticker = ticker,
        //                Price = (float)price,
        //                Time = DateTime.UtcNow // Or the appropriate time
        //            });
        //        }

        //        _GetDataTools.AddStockPrices(stockPrices);
        //        return Ok("Stock prices added successfully.");
        //    }
        //    catch (Exception ex)
        //    {
        //        // Log the exception
        //        return StatusCode(500, "Internal server error");
        //    }
        //}



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

        [HttpGet("/Home/AddFeaturedStock/{ticker}")]
        public IActionResult AddFeaturedStock(string ticker, string name)
        {
            try
            {
                _GetDataTools.AddStock(new Stock
                {
                    Ticker = ticker,
                    Name = name,
                    CreatedAt = DateTime.Now
                });
                return Ok("Stock added successfully.");
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }
        //Add user by sending url /Home/AddUser/?email={email}&password={password}
        [HttpPost("/Home/AddUser")]
        public IActionResult AddUser([FromBody] User user)
        {
            try
            {
                _GetDataTools.AddUser(new User
                {
                    Email = user.Email,
                    Password = user.Password, 
                    //UserTypeId = ?
                    CreatedAt = DateTime.Now 
                });
                    return Ok("User added successfully."); 
            } 
            catch (DbUpdateException ex) { 
                // Check if it's a unique constraint violation
                if (ex.InnerException is MySqlException sqlEx && sqlEx.Number == 1062) 
                { 
                    return Conflict($"Duplicate entry: This {user.Email} is already registered."); 
                } 
                else 
                {
                    return StatusCode(500, "Error in database"); 
                } 
            } catch (Exception ex) 
            { // Log the exception
              return StatusCode(500, "Internal server error");
            } 
        }

        [HttpGet("/Home/AddSeriesData/{ticker}")]
        public async Task<IActionResult> AddSeriesData(string ticker)
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


    }
}
