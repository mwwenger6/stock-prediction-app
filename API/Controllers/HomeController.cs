using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Primitives;
using MySqlConnector;
using Newtonsoft.Json;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.ViewModel;
using System;
using System.Diagnostics;
using System.Text.Json;
//using Python.Runtime;
using static Stock_Prediction_API.Services.API_Tools.TwelveDataTools;


namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config) : base(context, config) {}

        public DateTime GetEasternTime()
        {
            DateTime currentTime = DateTime.UtcNow;
            return TimeZoneInfo.ConvertTimeBySystemTimeZoneId(currentTime, TimeZoneInfo.Utc.Id, "Eastern Standard Time");
        }


        #region User

        [HttpGet("/Home/GetUsers")]
        public IActionResult GetUsers()
        {
            try
            {
                List<User> users = _GetDataTools.GetUsers().ToList();
                return Json(users);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error getting users.");
            }
        }

        [HttpGet("/Home/GetUser/{email}")]
        public IActionResult GetUser(string email)
        {
            try
            {
                User user = _GetDataTools.GetUser(email);
                user.TypeName = _GetDataTools.GetUserTypes().Single(t => t.Id == user.TypeId).UserTypeName;
                return Json(user);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }
        [HttpGet("/Home/AuthenticateUser/{email}/{password}")]
        public IActionResult AuthenticateUser(string email, string password)
        {
            try
            {
                User user = _GetDataTools.GetUser(email);
                user.TypeName = _GetDataTools.GetUserTypes().Single(t => t.Id == user.TypeId).UserTypeName;
                if (user.Password != password)
                    throw new InvalidDataException("Could not authenticate");
                return Json(user);
            }
            catch (InvalidDataException ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(401, ex.Message);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        //Add user by sending url /Home/AddUser/?email={email}&password={password}
        [HttpPost("/Home/AddUser")]
        public IActionResult AddUser([FromBody] User user)
        {
            try
            {
                int clientId = _GetDataTools.GetUserTypes().Single(t => t.UserTypeName == UserType.CLIENT).Id;
                _GetDataTools.AddUser(new User
                {
                    Email = user.Email,
                    Password = user.Password,
                    TypeId = clientId,
                    CreatedAt = GetEasternTime()
                });
                return Ok("User added successfully.");
            }
            catch (DbUpdateException ex)
            {
                // Check if it's a unique constraint violation
                if (ex.InnerException is MySqlException sqlEx && sqlEx.Number == 1062)
                {
                    return Conflict($"Duplicate entry: This {user.Email} is already registered.");
                }
                else
                {
                    _GetDataTools.LogError(new()
                    {
                        Message = ex.Message,
                        CreatedAt = GetEasternTime(),
                    });
                    return StatusCode(500, "Error in database");
                }
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        #endregion

        #region Stocks

        #region Prices
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        //Called by the background service every 5 mins
        [HttpPost("/Home/AddRecentStockPrices")]
        public async Task<IActionResult> AddRecentStockPrices()
        {
            try
            {
                DateTime dateTime = GetEasternTime();
                if (!(dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday &&
                   dateTime.Hour >= 9 && dateTime.Hour <= 15 && (dateTime.Hour != 9 || dateTime.Minute >= 30)))
                    return Ok("Market closed, no prices updated");

                List<string> tickers = _GetDataTools.GetStocks().Select(s => s.Ticker).ToList();
                List<StockPrice> stockList = new();
                foreach (string ticker in tickers)
                {
                    Stock stock = await _FinnhubDataTools.GetRecentPrice(ticker);
                    if (stock != null)
                    {
                        Console.WriteLine(ticker + ": " + stock.CurrentPrice + ", " + stock.DailyChange);
                        stockList.Add(new StockPrice
                        {
                            Ticker = ticker,
                            Price = (float)stock.CurrentPrice,
                            Time = dateTime,

                        });
                        _GetDataTools.UpdateStockPrice(new()
                        {
                            Ticker = ticker,
                            CurrentPrice = stock.CurrentPrice,
                            UpdatedAt = dateTime,
                            DailyChange = stock.DailyChange
                        });
                    }
                }
                //Add closing price to table if market is about to close
                if (dateTime.Hour == 15 && dateTime.Minute >= 55)
                    _GetDataTools.AddStockPrices(stockList);

                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        #endregion


        [HttpGet("/Home/GetStocks")]
        public IActionResult GetStocks()
        {
            try
            {
                List<Stock> stocks = _GetDataTools.GetStocks().ToList();
                return Json(stocks);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new ()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error getting stocks.");
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        [HttpPost("/Home/AddStock/{ticker}/{name}")]
        public IActionResult AddStock(string ticker, string name)
        {
            DateTime dateTime = GetEasternTime();
            try
            {
                _GetDataTools.AddStock(new Stock
                {
                    Ticker = ticker,
                    Name = name,
                    CreatedAt = dateTime,
                    UpdatedAt = dateTime
                });
                return Ok("Stock added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = dateTime,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        #endregion

        #region WatchList
        [HttpPost("/Home/AddUserWatchlistStock/{userId}/{ticker}")]
        public IActionResult AddUserWatchlistStock(int userId, string ticker)
        {
            try
            {
                //check that stock is in our db
                _GetDataTools.GetStock(ticker);

                //add stock
                _GetDataTools.AddUserWatchlistStock(new UserWatchlistStocks
                {
                    UserId = userId,
                    Ticker = ticker,
                });

                return Ok("User watchlist stock added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        [HttpPost("/Home/RemoveUserWatchlistStock/{userId}/{ticker}")]
        public IActionResult RemoveUserWatchlistStock(int userId, string ticker)
        {
            try
            {
                _GetDataTools.RemoveUserWatchlistStock(userId, ticker);

                return Ok("User removed watchlist stock successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

        [HttpGet("/Home/GetUserWatchlistStocks/{userId}")]
        public IActionResult GetUserWatchlistStocks(int userId)
        {
            try
            {
                List<UserWatchlistStocks> watchlistStocks = _GetDataTools.GetUserWatchlistStocks(userId);
                List<StockPrice> stocks = new List<StockPrice>();

                foreach (UserWatchlistStocks userStock in watchlistStocks)
                {
                    stocks.Add(_GetDataTools.GetRecentStockPrice(userStock.Ticker));
                }

                return Json(stocks);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }


        #endregion

        #region Admin

        [HttpGet("/Home/GetErrorLogs")]
        public IActionResult GetErrorLogs()
        {
            try
            {
                List<ErrorLog> errors = _GetDataTools.GetErrorLogs().ToList();
                return Json(errors);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error getting error logs.");
            }
        }

        [HttpPost("/Home/ChangeUserType/{email}/{userTypeName}")]
        public IActionResult ChangeUserType(string email, string userTypeName)
        {
            try
            {
                //Check that user type name is in db, get id of that type
                List <UserType> userTypes = _GetDataTools.GetUserTypes().ToList();
                int newTypeId = userTypes.Single(t => t.UserTypeName.ToLower() == userTypeName.ToLower()).Id;

                //Update the user's type id
                _GetDataTools.UpdateUserPrivileges(email, newTypeId);

                return Ok($"User type changed to {userTypeName}"); 
            }
            catch (InvalidOperationException)
            {
                return BadRequest("Invalid user type name");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }


        #endregion

        #region Prediction
        [HttpGet("/Home/TrainModel/{ticker}")]
        public IActionResult TrainModel(string ticker)
        {
            //string tempFilePath = Path.Combine("PythonScripts", "tempJsonFile.txt");
            try
            {
                //List<StockPrice> historicalData = _GetDataTools.GetStockPrices(ticker).ToList();
                //var options = new JsonSerializerOptions { WriteIndented = true };
                //var historicalDataJson = System.Text.Json.JsonSerializer.Serialize(historicalData, options);

                //// Write the JSON data to a temporary file
                //System.IO.File.WriteAllText(tempFilePath, historicalDataJson);

                string pythonScriptPath = Path.Combine("PythonScripts", "model_train.py");
                ProcessStartInfo start = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScriptPath}\" --ticker \"{ticker}\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (Process process = Process.Start(start))
                {
                    using (StreamReader reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        process.WaitForExit();

                        //// Optionally delete the temp file if it's no longer needed
                        //System.IO.File.Delete(tempFilePath);

                        return Content(result);
                    }
                }
            }
            catch (Exception ex)
            {
                //// Optionally delete the temp file in case of an exception
                //System.IO.File.Delete(tempFilePath);
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }


        [HttpGet("/Home/Predict/{ticker}/{prediction_range}")]
        public IActionResult Predict(string ticker, int prediction_range)
        {
            try
            {
                // Pass the JSON data to the Python script
                string pythonScriptPath = "wwwroot\\PythonScripts\\model_predict.py";
                ProcessStartInfo start = new()
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScriptPath}\" --ticker \"{ticker}\" --range \"{prediction_range}\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardError = true
                };
#nullable disable
                using Process process = Process.Start(start);
                using StreamReader reader = process.StandardOutput;
                string errors = process.StandardError.ReadToEnd();
                if (!string.IsNullOrEmpty(errors))
                {
                    return StatusCode(500, errors);
                }
                string result = reader.ReadToEnd();
                process.WaitForExit();
                return Ok(result);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }
        #endregion
    }
}
