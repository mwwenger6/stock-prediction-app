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
using Python.Runtime;
using static Stock_Prediction_API.Services.API_Tools.TwelveDataTools;


namespace Stock_Prediction_API.Controllers
{
    public class HomeController : ControllerHelper
    {
        public HomeController(AppDBContext context, IConfiguration config) : base(context, config) {}

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
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Error getting users.");
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
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Error getting stocks.");
            }                  
        }
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
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Error getting error logs.");
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
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
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
                if (user.Password != password)
                    throw new InvalidDataException("Could not authenticate");
                return Json(user);
            }
            catch (InvalidDataException ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(401, ex.Message);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
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
                    CreatedAt = DateTime.Now,
                    UpdatedAt = DateTime.Now
                });
                return Ok("Stock added successfully.");
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
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
                    CreatedAt = DateTime.Now,
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
                    _GetDataTools.LogError(new()
                    {
                        Message = ex.Message,
                        CreatedAt = DateTime.Now,
                    });
                    return StatusCode(500, "Error in database"); 
                } 
            } catch (Exception ex) 
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
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
                DateTime currentTime = DateTime.UtcNow;
                DateTime dateTime = TimeZoneInfo.ConvertTimeBySystemTimeZoneId(currentTime, TimeZoneInfo.Utc.Id, "Eastern Standard Time");

                if (!(dateTime.DayOfWeek >= DayOfWeek.Monday && dateTime.DayOfWeek <= DayOfWeek.Friday &&
                   dateTime.Hour >= 9 && dateTime.Hour < 24 && (dateTime.Hour != 9 || dateTime.Minute >= 30)))
                    return Ok("Market closed, no prices updated");

                List<string> tickers = _GetDataTools.GetStocks().Select(s => s.Ticker).ToList();
                List<StockPrice> stockList = new();
                foreach (string ticker in tickers)
                {
                    Stock stock = await _FinnhubDataTools.GetRecentPrice(ticker);
                    if(stock != null)
                    {
                        Console.WriteLine(ticker + ": " + stock.CurrentPrice + ", " + stock.DailyChange);
                        stockList.Add(new StockPrice
                        {
                            Ticker = ticker,
                            Price = (float)stock.CurrentPrice,
                            Time = DateTime.Now,

                        });
                        _GetDataTools.UpdateStockPrice(new()
                        {
                            Ticker = ticker,
                            CurrentPrice = stock.CurrentPrice,
                            UpdatedAt = DateTime.Now,
                            DailyChange = stock.DailyChange
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
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
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
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }
                

        //[HttpPost("/Home/AddHistoricStockData/{ticker}/{interval}/{outputSize}")]
        //public async Task<IActionResult> AddHistoricStockData(string ticker, string interval, string outputSize)
        //{
        //    try
        //    {
        //        List<StockPrice> stockPrices = await _TwelveDataTools.GetTimeSeriesData(ticker, interval, outputSize);
        //        _GetDataTools.AddStockPrices(stockPrices);

        //        return Ok("Stock prices added successfully.");
        //    }
        //    catch (Exception ex)
        //    {
        //        // Log the exception
        //        return StatusCode(500, "Internal server error");
        //    }
        //}

        [HttpGet("/Home/TestPythonScript")]
        public IActionResult TestPythonScript()
        {
            string pythonScriptPath = Path.Combine("PythonScripts", "samplePython.py");
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = "python",  // Ensure that python is in your PATH or provide the full path to the python executable
                Arguments = $"\"{pythonScriptPath}\"",
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
                    return Content(result);
                }
            }
        }

        [HttpGet("/Home/TrainsModel/{ticker}")]
        public IActionResult TrainsModel(string ticker)
        {
            try
            {
                List<StockPrice> historicalData = _GetDataTools.GetStockPrices(ticker).ToList();
                var options = new JsonSerializerOptions { WriteIndented = true };
                var historicalDataJson = System.Text.Json.JsonSerializer.Serialize(historicalData, options);
                //ProcessStartInfo ProcessInfo = new("python3")
                //{
                //    UseShellExecute = false,
                //    RedirectStandardOutput = true,
                //    Arguments = Path.Combine("PythonScripts", "model_train.py")
                //};
                //Process process = new()
                //{
                //    StartInfo = ProcessInfo
                //};
                //process.Start();
                //StreamReader reader = process.StandardOutput;
                //String line = reader.ReadLine();
                //process.WaitForExit();
                //process.Close();

                //ProcessStartInfo start = new ProcessStartInfo
                //{
                //    FileName = "python",
                //    Arguments = $"\"{pythonScriptPath}\" --jsonFile \"{tempFilePath}\" --ticker \"{ticker}\"",
                //    RedirectStandardOutput = true,
                //    UseShellExecute = false,
                //    CreateNoWindow = true
                //};

                //using (Process process = Process.Start(start))
                //{
                //    using (StreamReader reader = process.StandardOutput)
                //    {
                //        string result = reader.ReadToEnd();
                //        process.WaitForExit();
                //        return Content(result);
                //    }
                //}
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
            return StatusCode(500, "");
        }

        [HttpGet("/Home/TrainModel/{ticker}")]
        public IActionResult TrainModel(string ticker)
        {
            string tempFilePath = Path.Combine("PythonScripts", "tempJsonFile.txt");
            try
            {
                List<StockPrice> historicalData = _GetDataTools.GetStockPrices(ticker).ToList();
                var options = new JsonSerializerOptions { WriteIndented = true };
                var historicalDataJson = System.Text.Json.JsonSerializer.Serialize(historicalData, options);

                // Write the JSON data to a temporary file
                System.IO.File.WriteAllText(tempFilePath, historicalDataJson);

                string pythonScriptPath = Path.Combine("PythonScripts", "model_train.py");
                ProcessStartInfo start = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScriptPath}\" --jsonFile \"{tempFilePath}\" --ticker \"{ticker}\"",
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

                        // Optionally delete the temp file if it's no longer needed
                        System.IO.File.Delete(tempFilePath);

                        return Content(result);
                    }
                }
            }
            catch (Exception ex)
            {
                // Optionally delete the temp file in case of an exception
                System.IO.File.Delete(tempFilePath);
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
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
                return Content(result);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = DateTime.Now,
                });
                return StatusCode(500, $"Internal server error. {ex.Message}");
            }
        }

    }
}
