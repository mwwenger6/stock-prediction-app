using Microsoft.AspNetCore.Mvc;
using Microsoft.Identity.Client;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.ViewModel;
using System.Diagnostics;
using System.Globalization;


namespace Stock_Prediction_API.Controllers
{
    public class StockController : ControllerHelper
    {
        public StockController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        [HttpGet("/Stock/GetHistoricalStockData/{ticker}")]
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

        //Ticker must be featured stock, startDate must be in form mmddyyy, interval must be 5min or 1day
        [HttpGet("/Stock/GetStockGraphData/{ticker}/{startDate}/{interval}")]
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

                List<StockPrice> stockPrices = _GetDataTools.GetStockPrices(ticker, getHistoricalData, parsedStartDate).ToList();

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

        [HttpGet("/Stock/GetUserWatchlistStocks/{userId}")]
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

        [HttpGet("/Stock/GetSupportedStocks")]
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

        [HttpGet("/Stock/UpdateSupportedStocks")]
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

        [HttpGet("/Stock/GetStocks")]
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

        [HttpGet("/Stock/GetStock/{ticker}")]
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

        class TechnicalStockInfo
        {
            public string Ticker { get; set; }
            public string Name { get; set; }
            public float MeanPercentReturn { get; set; }
            public float PercentVolatility { get; set; }
            public List<float> PricePoints { get; set; }
        }
        [HttpGet("/Stock/GetTechnicalStockInfo/{numDaysLookback}/{useClosePrices}/")]
        public IActionResult GetTechnicalStockInfo(int numDaysLookback, bool useClosePrices)
        {

            try
            {
                List<TechnicalStockInfo> technicalInfo = new();
                _GetDataTools.GetStocks().ToList().ForEach(stock =>
                {
                    //Get prices
                    DateTime oneWeekAgo = DateTime.Now.Subtract(TimeSpan.FromDays(numDaysLookback));
                    List<StockPrice> currStockPrices = _GetDataTools.GetStockPrices(stock.Ticker, useClosePrices, oneWeekAgo).ToList();
                    List<float> prices = currStockPrices.Select(x => x.Price).ToList();

                    //Calculate returns
                    List<float> returns = new();
                    for (int i = 0; i < prices.Count - 1; i++)
                    {
                        float ret = (prices[i] - prices[i + 1]) / prices[i + 1];
                        ret = (float)Math.Round(ret, 5);
                        returns.Add(ret);
                    }

                    //Calculate volatility
                    float meanReturn = returns.Average();
                    float sumSquaredDifferences = (float)returns.Sum(ret => Math.Pow(ret - meanReturn, 2));
                    float variance = sumSquaredDifferences / returns.Count;
                    float volatility = (float)Math.Sqrt(variance);
                    technicalInfo.Add(new()
                    {
                        Name = stock.Name,
                        Ticker = stock.Ticker,
                        MeanPercentReturn = meanReturn * 100,
                        PercentVolatility = volatility * 100,
                        PricePoints = prices
                    });
                });
                technicalInfo = technicalInfo.OrderByDescending(x => x.MeanPercentReturn).ToList();

                return Json(technicalInfo);
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Problem getting technical stock info. {ex.Message}");
            }
        }

        [HttpGet("/Stock/GetTechnicalStockInfoForStock/{numDaysLookback}/{useClosePrices}/{ticker}")]
        public IActionResult GetTechnicalStockInfoForStock(int numDaysLookback, bool useClosePrices, string ticker)
        {

            try
            {
                    //Make sure stock exists
                    Stock featuredStock = _GetDataTools.GetStock(ticker);
                    //Get prices
                    DateTime oneWeekAgo = DateTime.Now.Subtract(TimeSpan.FromDays(numDaysLookback));
                    List<StockPrice> currStockPrices = _GetDataTools.GetStockPrices(ticker, useClosePrices, oneWeekAgo).ToList();
                    List<float> prices = currStockPrices.Select(x => x.Price).ToList();

                    //Calculate returns
                    List<float> returns = new();
                    for (int i = 0; i < prices.Count - 1; i++)
                    {
                        float ret = (prices[i] - prices[i + 1]) / prices[i + 1];
                        ret = (float)Math.Round(ret, 5);
                        returns.Add(ret);
                    }

                    //Calculate volatility
                    float meanReturn = returns.Average();
                    float sumSquaredDifferences = (float)returns.Sum(ret => Math.Pow(ret - meanReturn, 2));
                    float variance = sumSquaredDifferences / returns.Count;
                    float volatility = (float)Math.Sqrt(variance);
                    return Json(new TechnicalStockInfo
                    {
                        Name = featuredStock.Name,
                        Ticker = featuredStock.Ticker,
                        MeanPercentReturn = meanReturn * 100,
                        PercentVolatility = volatility * 100,
                        PricePoints = prices
                    });
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Problem getting technical stock info. {ex.Message}");
            }
        }

        [HttpGet("/Stock/GetOpenMarketDays/{numDays}")]
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

        //Called by web job every 5 mins
        [HttpPost("/Stock/AddRecentStockPrices")]
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
                    bool closePrices = false;
                    if (dateTime.Hour == 15 && dateTime.Minute >= 55)
                        closePrices = true;

                    if (stock != null)
                    {
                        //Add to the list of stock prices
                        stockList.Add(new()
                        {
                            Ticker = ticker,
                            Price = (float)stock.CurrentPrice,
                            Time = dateTime,
                            IsClosePrice = closePrices
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
                return StatusCode(500, $"Error when adding recent stock prices to DB. {ex.Message}");
            }
        }

        [HttpPost("/Stock/AddStock/{ticker}/{name}")]
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

        //Can add time series data: either 5min or 1day. 5min data does not add the close prices because we do not want
        //extra prices. Thus if adding a new stock, first add ~ 2500 1day prices and ~5000 5min prices
        [HttpPost("/Stock/AddHistoricStockData/{ticker}/{interval}/{outputSize}")]
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
                if(interval == "5min")
                {
                    stockPrices.RemoveAll(price => price.Time.TimeOfDay == new TimeSpan(15, 55, 0));
                }

                if (interval == "1day")
                {
                    foreach(StockPrice price in stockPrices)
                    {
                        price.IsClosePrice = true;
                    }
                }
                _GetDataTools.AddStockPrices(stockPrices);

                
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

        [HttpPost("/Stock/RemoveStock/{ticker}")]
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


        [HttpPost("/Stock/AddUserWatchlistStock/{userId}/{ticker}")]
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

        [HttpPost("/Stock/RemoveUserWatchlistStock/{userId}/{ticker}")]
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


    }
}
