using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Primitives;
using MySqlConnector;
using Newtonsoft.Json;
using ServiceStack;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.ViewModel;
using System;
using System.Diagnostics;
using System.Text.Json;
//using Python.Runtime;
using static Stock_Prediction_API.Services.API_Tools.TwelveDataTools;
using BCrypt.Net;
using System.Globalization;


namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

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
                return StatusCode(500, $"Error Getting User. {ex.Message}");
            }
        }

        [HttpGet("/Home/AuthenticateUser/{email}/{password}")]
        public IActionResult AuthenticateUser(string email, string password)
        {
            try
            {
                User user = _GetDataTools.GetUser(email);
                
                if (user == null)
                {
                    return StatusCode(404, "User not found.");
                }
        
                user.TypeName = _GetDataTools.GetUserTypes().Single(t => t.Id == user.TypeId).UserTypeName;

                if (!BCrypt.Net.BCrypt.Verify(password, user.Password))
                {
                    return StatusCode(401, "Invalid credentials.");
                }

                if (!user.IsVerified)
                {
                    return StatusCode(403, "User not verified.");
                }
        
                return Json(user);
            }
            catch (InvalidDataException ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = $"Authentication error for {email}: {ex.Message}",
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(401, ex.Message);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = $"Unexpected error during authentication for {email}: {ex.Message}",
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error authenticating user. {ex.Message}");
            }
        }


        [HttpPost("/Home/DeleteUser/{email}")]
        public IActionResult DeleteUser(string email)
        {
            try
            {
                _GetDataTools.DeleteUser(email);
                return Ok("User is deleted");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not delete user. {ex.Message}");
            }
        }

        //Add user by sending url /Home/AddUser/?email={email}&password={password}
        [HttpPost("/Home/AddUser")]
        public IActionResult AddUser([FromBody] User user)
        {
            string allowedChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
            int length = 12;

            Random random = new();
            string randomString = new(Enumerable.Repeat(allowedChars, length)
                .Select(s => s[random.Next(s.Length)]).ToArray());
            try
            {
                int clientId = _GetDataTools.GetUserTypes().Single(t => t.UserTypeName == UserType.CLIENT).Id;
                _GetDataTools.AddUser(new User
                {
                    Email = user.Email,
                    Password = user.Password,
                    TypeId = clientId,
                    CreatedAt = GetEasternTime(),
                    IsVerified = false,
                    VerificationCode = randomString
                });
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
                return StatusCode(500, $"Error Adding User to DB. {ex.Message}");
            }
            try
            {
                _EmailTools.SendVerificationEmail(user.Email, randomString);
                return Ok("User added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error Sending Email. {ex.Message}");
            }
        }

        [HttpPost("/Home/VerifyUser/{code}")]
        public IActionResult VerifyUser(string code)
        {
            try
            {
                if (_GetDataTools.UserWithVerificationCode(code))
                {
                    User user = _GetDataTools.GetUserByVerificationCode(code);
                    user.IsVerified = true;
                    user.VerificationCode = null;
                    _GetDataTools.AddUser(user);
                    return Ok("User Verified.");
                }
                else
                {
                    return StatusCode(500, $"No User to be verified. If this is not true contact support.");
                }
            }
            catch(Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error With Verifying User. {ex.Message}");
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
                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker, true).ToList();
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
                return StatusCode(500, $"Problem getting historical stock prices from DB. {ex.Message}");
            }
        }

        [HttpGet("/Home/Get5MinStockData/{ticker}")]
        public IActionResult Get5MinStockData(string ticker)
        {
            try
            {
                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker, false).ToList();
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
                return StatusCode(500, $"Problem getting 5 minute data for {ticker}. {ex.Message}");
            }
        }

        //Ticker must be featured stock, startDate must be in form mmddyyy, interval must be 5min or 1day
        [HttpGet("/Home/GetStockGraphData/{ticker}/{startDate}/{interval}")]
        public IActionResult GetStockGraphData(string ticker, string startDate, string interval)
        {
            try
            {
                if (interval != "5min" && interval != "1day")
                    return StatusCode(500, $"Invalid interval");

                bool getHistoricalData = interval == "1day";

                // Parse the startDate string to a DateTime object
                DateTime parsedStartDate;
                if (!DateTime.TryParseExact(startDate, "MMddyyyy", CultureInfo.InvariantCulture, DateTimeStyles.None, out parsedStartDate))
                    return StatusCode(500, $"Invalid startDate format");

                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker, getHistoricalData)
                    .Where(s => s.Time >= parsedStartDate)
                    .ToList();

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
                return StatusCode(500, $"Problem getting 5 minute data for {ticker}. {ex.Message}");
            }
        }


        //Called by the background service every 5 mins
        [HttpPost("/Home/AddRecentStockPrices")]
        public async Task<IActionResult> AddRecentStockPrices()
        {
            try
            {
                DateTime dateTime = GetEasternTime();
                //Check if market is open
                bool? isOpen = _FinnhubDataTools.GetIsMarketOpen().Result;
                if (isOpen == null)
                    throw new Exception("Unable to retrieve market status");
                else if (!(bool)isOpen)
                    return Ok("Market closed, no prices updated");

                List<string> tickers = _GetDataTools.GetStocks().Select(s => s.Ticker).ToList();
                List<StockPrice> stockList = new();
                foreach (string ticker in tickers)
                {
                    Stock stock = await _FinnhubDataTools.GetRecentPrice(ticker);
                    if (stock != null)
                    {
                        //Add to the list of stock prices
                        stockList.Add(new()
                        {
                            Ticker = ticker,
                            Price = (float)stock.CurrentPrice,
                            Time = dateTime,

                        });
                        //Update in stock table
                        _GetDataTools.UpdateStockPrice(new()
                        {
                            Ticker = ticker,
                            CurrentPrice = stock.CurrentPrice,
                            UpdatedAt = dateTime,
                            DailyChange = stock.DailyChange
                        });
                    }
                }
                //Update in 5 min interval table
                _GetDataTools.AddFMStockPrices(stockList);
                //Add closing price to table if market is about to close
                if (dateTime.Hour == 15 && dateTime.Minute >= 55)
                    _GetDataTools.AddEODStockPrices(stockList);

                return Ok("Stock prices added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error when adding recent stock prices to DB. {ex.Message}");
            }
        }
        [HttpPost("/Home/AddHistoricStockData/{ticker}/{interval}/{outputSize}")]
        public async Task<IActionResult> AddHistoricStockData(string ticker, string interval, string outputSize)
        {
            try
            {
                interval = interval.ToLower();
                int outputSizeNum = int.Parse(outputSize);
                //Check paramaters
                if (outputSizeNum > 5000 || outputSizeNum < 0)
                    throw new Exception("Output size is limited to 5000");
                if (interval != "5min" && interval != "1day")
                    throw new Exception("Supported intervals are '5min' and '1day'");
                try
                {
                    _GetDataTools.GetStock(ticker);
                }
                catch(InvalidOperationException ex)
                {
                    throw new Exception("Stock not found in database. Requested stock must be a StockGenie featured stock");
                }

                List<Stock> list = _GetDataTools.GetStocks().ToList();

                List<StockPrice> stockPrices = await _TwelveDataTools.GetTimeSeriesData(ticker, interval, outputSize);

                if(interval == "1day")
                    _GetDataTools.AddEODStockPrices(stockPrices);
                else
                    _GetDataTools.AddFMStockPrices(stockPrices);

                
                return Ok($"{outputSize} {ticker} stock prices added successfully for at {interval} intervals.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Problem adding historical data to DB. {ex.Message}");
            }
        }

        #endregion

        [HttpGet("/Home/GetSupportedStocks")]
        public IActionResult GetSupportedStocks()
        {
            try
            {
                List<SupportedStocks> stocks = _GetDataTools.GetSupportedStocks();
                return Json(stocks);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Problem getting Supported Stocks from DB.");
            }
        }

        [HttpGet("/Home/UpdateSupportedStocks")]
        public IActionResult UpdateSupportedStocks()
        {
            try
            {
                List<SupportedStocks> stocks = _TwelveDataTools.GetSupportedStockData().Result;
                int addedCount = stocks.Count();
                int removedCount = _GetDataTools.RemoveSupportedStocks();
                _GetDataTools.AddSupportedStocks(stocks);
                return Json(removedCount + " stocks removed and " + addedCount + " stocks added.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Problem updating Supported Stocks.");
            }
        }

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
                return StatusCode(500, $"Problem getting Stocks from DB.");
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
                return StatusCode(500, $"Problem getting stock {ticker}. {ex.Message}");
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
                return StatusCode(500, $"Could not add stock {ticker}. {ex.Message}");
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
                return StatusCode(500, $"Could not remove stock {ticker}. {ex.Message}");
            }
        }

        [HttpGet("/Home/GetOpenMarketDays/{numDays}")]
        public IActionResult GetOpenMarketDays(int numDays)
        {
            if (numDays < 0 || numDays > 365) throw new ArgumentOutOfRangeException("Number of requested days out of valid range");

            List<DateTime> tradingDays = new List<DateTime>();
            List<DateTime> holidays = _GetDataTools.GetMarketHolidays().Select(holiday => holiday.Day).ToList(); ;
            DateTime currentDate = DateTime.Now.Date;

            for (int i = 0; i < numDays; i++)
            {
                // Moves currentDate to the next open market day if current day is a weekend or holiday
                // Note that the holidays are stored in ascending by present to future, so I can just check the first element of the list and pop when they match
                while (currentDate.DayOfWeek == DayOfWeek.Saturday || currentDate.DayOfWeek == DayOfWeek.Sunday || holidays.Contains(currentDate))
                {
                    currentDate = currentDate.AddDays(1);
                }
                tradingDays.Add(currentDate);
                currentDate = currentDate.AddDays(1); // Move to the next day
            }

            return Json(tradingDays);
        }

        private void StockEmailHelper(string email, string ticker, bool isIncreasing)
        {
            StockEmailViewModel vm = new()
            {
                Email = email,
                Ticker = ticker,
                Indication = isIncreasing ? "RISE" : "DECLINE",
                StockName = _GetDataTools.GetStock(ticker).Name,
            };
            _EmailTools.SendStockEmail(vm);
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
                return StatusCode(500, $"Could not add stock to watch list; stock: {ticker}, user: {userId}. {ex.Message}");
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
                return StatusCode(500, $"Problem removing {ticker} from user's watch list; User: {userId}. {ex.Message}");
            }
        }

        [HttpGet("/Home/GetUserWatchlistStocks/{userId}")]
        public IActionResult GetUserWatchlistStocks(int userId)
        {
            try
            {
                List<UserWatchlistStocks> watchlistStocks = _GetDataTools.GetUserWatchlistStocks(userId);
                List<Stock> stocks = new List<Stock>();

                foreach (UserWatchlistStocks userStock in watchlistStocks)
                {
                    stocks.Add(_GetDataTools.GetStock(userStock.Ticker));
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
                return StatusCode(500, $"Problem getting user's watchlist stocks; User: {userId}. {ex.Message}");
            }
        }


        #endregion

        #region User Stocks

        [HttpGet("/Home/GetUserStocks/{userId}")]
        public IActionResult GetUserStocks(int userId)
        {
            try
            {
                return Json(_GetDataTools.GetUserStocks(userId).ToList());
            }
            catch(Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not get user stocks: {userId}. {ex.Message}");
            }
        }

        [HttpGet("/Home/GetUserStockData/{userId}")]
        public IActionResult GetUserStockData(int userId)
        {
            List<UserStock> userStocks = new();
            float[]? userStockPrices = null;
            try
            {
                userStocks = _GetDataTools.GetUserStocks(userId).ToList();
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not get user stocks: {userId}. {ex.Message}");
            }
            foreach(UserStock userStock in userStocks)
            {

                List<StockPrice> prices = new();
                try
                {
                    prices = _GetDataTools.GetStockPrices(userStock.Ticker, DateTime.Now.Date).ToList();
                }
                catch (Exception ex)
                {
                    _GetDataTools.LogError(new()
                    {
                        Message = ex.Message,
                        CreatedAt = GetEasternTime(),
                    });
                    return StatusCode(500, $"Could not get stock prices for stock: {userStock.Ticker}. {ex.Message}");
                }
                int count = 0;
                userStockPrices ??= new float[prices.Count];
                foreach(StockPrice stockPrice in prices)
                {
                    userStockPrices[count] = stockPrice.Price * userStock.Quantity;
                    count++;
                }
            }
            return Json(userStockPrices);
        }

        [HttpPost("/Home/AddUserStock/{userId}/{ticker}/{quantity}")]
        public IActionResult AddUserStock(int userId, string ticker, float quantity)
        {
            try
            {
                _GetDataTools.AddUserStock(new UserStock
                {
                    UserId = userId,
                    Ticker = ticker,
                    Quantity = quantity,
                    CreatedAt = DateTime.Now
                });
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not add Stock to User: {userId}. {ex.Message}");
            }
            return Ok("Stock Added Successfully.");
        }

        [HttpPost("/Home/RemoveUserStock/{userId}/{ticker}")]
        public IActionResult RemoveUserStock(int userId, string ticker)
        {
            try
            {
                _GetDataTools.RemoveUserStock(new UserStock
                {
                    UserId = userId,
                    Ticker = ticker,
                });
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not remove Stock for User: {userId}. {ex.Message}");
            }
            return Ok("Stock Removed Successfully.");
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
                return StatusCode(500, $"Problem getting error logs.");
            }
        }

        [HttpPost("/Home/DeleteErrorLog/{id}")]
        public IActionResult DeleteErrorLog(int id)
        {
            try
            {
                var log = _GetDataTools.GetErrorLogs().FirstOrDefault(e => e.Id == id);
                if (log == null)
                {
                    return NotFound($"Error log with ID {id} not found.");
                }

                _GetDataTools.DeleteErrorLog(log); // Assume this method deletes the log
                return Ok();
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new ErrorLog
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.UtcNow // Adjust as needed
                });
                return StatusCode(500, "Problem deleting the error log.");
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
                return StatusCode(500, $"Could not change user type. {ex.Message}");
            }
        }

        [HttpPost("/Home/AddMarketHolidays")]
        public IActionResult AddMarketHolidays()
        {
            try
            {
                List<MarketHolidays> holidays = _FinnhubDataTools.GetMarketHolidays().Result;
                _GetDataTools.AddMarketHolidays(holidays);
                return Ok(holidays.Count() + " stock market holidays added successfully");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not add market holiday. {ex.Message}");
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
                return StatusCode(500, $"Problem training model for {ticker}. {ex.Message}");
            }
        }

        public float[] Predict(string ticker, int prediction_range)
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
                    _GetDataTools.LogError(new()
                    {
                        Message = errors,
                        CreatedAt = GetEasternTime(),
                    });
                    return null;
                }
                string result = reader.ReadToEnd();
                process.WaitForExit();
                string[] parts = result.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                float[] predictions = Array.ConvertAll(parts, float.Parse);
                return predictions;
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return null;
            }
        }

        [HttpPost("/Home/ClearPredictions")]
        public IActionResult ClearPredictions()
        {
            _GetDataTools.ClearStockPredictions();
            return Ok("Predictions removed successfully.");
        }

        [HttpPost("/Home/AddPredictions/{ticker}")]
        public IActionResult AddPrediction(string ticker)
        {
            try
            {
                //DateTime dateTime = GetEasternTime();
                //if (!(dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday &&
                //   dateTime.Hour >= 9 && dateTime.Hour <= 15 && (dateTime.Hour != 9 || dateTime.Minute >= 30)))
                //    return Ok("Market closed, no new predictions");
                List<StockPrediction> batchPredictions = new();
                float[] predictions = Predict(ticker, 90);
                if (predictions != null)
                {
                    int order = 1;
                    foreach (float prediction in predictions)
                    {
                        batchPredictions.Add(new StockPrediction
                        {
                            Ticker = ticker,
                            PredictedPrice = prediction,
                            PredictionOrder = order,
                        });
                        order++;
                    }
                }
                _GetDataTools.AddPredictions(batchPredictions);

                return Ok($"Prediction for {ticker} added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Internal server error for adding prediction for ticker: {ticker}. {ex.Message}");
            }
        }

        //[HttpPost("/Home/AddPredictions")]
        //public IActionResult AddPredictions()
        //{
        //    try
        //    {
        //        //DateTime dateTime = GetEasternTime();
        //        //if (!(dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday &&
        //        //   dateTime.Hour >= 9 && dateTime.Hour <= 15 && (dateTime.Hour != 9 || dateTime.Minute >= 30)))
        //        //    return Ok("Market closed, no new predictions");

        //        _GetDataTools.ClearStockPredictions();
        //        List<string> tickers = _GetDataTools.GetStocks().Select(s => s.Ticker).ToList();
        //        List<StockPrediction> batchPredictions = new();
        //        foreach (string ticker in tickers)
        //        {
        //            float[] predictions = Predict(ticker, 90);
        //            if (predictions != null)
        //            {
        //                int order = 1;
        //                foreach (float prediction in predictions)
        //                {
        //                    batchPredictions.Add(new StockPrediction
        //                    {
        //                        Ticker = ticker,
        //                        PredictedPrice = prediction,
        //                        PredictionOrder = order,
        //                    });
        //                    order++;
        //                }
        //            }
        //        }
        //        _GetDataTools.AddPredictions(batchPredictions);

        //        return Ok("Predictions added successfully.");
        //    }
        //    catch (Exception ex)
        //    {
        //        _GetDataTools.LogError(new()
        //        {
        //            Message = ex.Message,
        //            CreatedAt = GetEasternTime(),
        //        });
        //        return StatusCode(500, $"Internal server error. {ex.Message}");
        //    }
        //}

        [HttpGet("/Home/GetPredictions/{ticker}")]
        public IActionResult GetPredictions(string ticker)
        {
            try
            {
                List<float> predictions = _GetDataTools.GetStockPredictions(ticker).ToList();
                return Json(predictions);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error getting stocks.");
            }
        }

        #endregion
    }
}
