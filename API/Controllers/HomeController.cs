using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Primitives;
using MySqlConnector;
using Newtonsoft.Json;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.ViewModel;
using System;
using static Stock_Prediction_API.Services.API_Tools.TwelveDataTools;


namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config) : base(context, config) {}

        [HttpGet("/Home/GetData/{type}")]
        public IActionResult GetData(string type)
        {
            try
            {
                switch (type.ToLower())
                {
                    case "users":
                        List<User> users = _GetDataTools.GetUsers().ToList();
                        return Json(users);
                    case "quickstocks":
                        List<QuickStock> quickStocks = _GetDataTools.GetQuickStocks().ToList();
                        return Json(quickStocks);
                    case "stocks":
                        List<Stock> stocks = _GetDataTools.GetStocks().ToList();
                        return Json(stocks);
                    case "stockprices":
                        List<StockPrice> prices = _GetDataTools.GetStockPrices().ToList();
                        return Json(prices);
                    default:
                        // Handle invalid type
                        return BadRequest("Invalid type parameter.");
                }
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Error getting {type} data.");
            }
        }

        [HttpGet("/Home/GetStock/{ticker}")]
        public IActionResult GetStock(string ticker)
        {
            try
            {
                Stock stock = _GetDataTools.GetStock(ticker);
                if (stock == null)
                {
                    return NotFound("Stock price not found.");
                }

                return Json(stock);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }


        [HttpGet("/Home/GetHistoricalStockData/{ticker}")]
        public IActionResult GetHistoricalStockData(string ticker)
        {
            try
            {
                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker).ToList();
                if (stockPrices == null || !stockPrices.Any())
                {
                    return NotFound("Stock prices not found");
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


        [HttpPost("/Home/AddStock/{ticker}/{name}")]
        public IActionResult AddStock(string ticker, string name)
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
        [HttpPost("/Home/RemoveStock/{ticker}")]
        public IActionResult RemoveStock(string ticker)
        {
            try
            {
                _GetDataTools.RemoveStock(ticker);
                return Ok("Stock removed successfully.");
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

        //Called by the background service every 5 mins
        [HttpPost("/Home/AddRecentStockPrices")]
        public async Task<IActionResult> AddRecentStockPrices()
        {
            try
            {
                DateTime dateTime = DateTime.Now;
                if (!(dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday &&
                   dateTime.Hour >= 9 && dateTime.Hour < 16 && (dateTime.Hour != 9 || dateTime.Minute >= 30)))
                    return Ok("Market closed, no prices updated");

                List<string> tickers = _GetDataTools.GetStocks().Select(s => s.Ticker).ToList();
                List<StockPrice> stockList = new();
                foreach (string ticker in tickers)
                {
                    float price = await _FinnhubDataTools.GetRecentPrice(ticker);
                    if(price != -1)
                    {
                        Console.WriteLine(ticker + ": " + price);
                        stockList.Add(new StockPrice
                        {
                            Ticker = ticker,
                            Price = price,
                            Time = DateTime.Now
                        });
                        _GetDataTools.UpdateStockPrice(new()
                        {
                            Ticker = ticker,
                            CurrentPrice = price
                        });
                    }
                }
                //Add closing price to table if market is about to close
                if(dateTime.Hour == 15 && dateTime.Minute >= 55)
                    _GetDataTools.AddStockPrices(stockList);

                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("/Home/AddHistoricStockData/{ticker}/{interval}/{outputSize}")]
        public async Task<IActionResult> AddHistoricStockData(string ticker, string interval, string outputSize)
        {
            try
            {
                List<StockPrice> stockPrices = await _TwelveDataTools.GetTimeSeriesData(ticker, interval, outputSize);
                _GetDataTools.AddStockPrices(stockPrices);

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
