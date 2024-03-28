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
using Microsoft.AspNetCore.Mvc;

/// <summary>
/// Controller responsible for handling stock prediction-related operations, including training models, making predictions, and managing prediction data.
/// Inherits from ControllerHelper to utilize shared functionalities and services.
/// </summary>  
namespace Stock_Prediction_API.Controllers
{
    public class PredictionController : ControllerHelper
    {
        public PredictionController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        /// <summary>
        /// Initiates training of a prediction model for a specific stock ticker.
        /// </summary>
        /// <param name="ticker">The stock ticker symbol for which to train the model.</param>
        /// <returns>A result message indicating the training outcome.</returns>
        [HttpGet("/Prediction/TrainModel/{ticker}")]
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

        /// <summary>
        /// Generates predictions for a given stock ticker and prediction range.
        /// </summary>
        /// <param name="ticker">The stock ticker symbol for which to generate predictions.</param>
        /// <param name="prediction_range">The range in days for the prediction.</param>
        /// <returns>An array of predicted values.</returns>
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

        /// <summary>
        /// Retrieves stored predictions for a given stock ticker.
        /// </summary>
        /// <param name="ticker">The stock ticker symbol for which to retrieve predictions.</param>
        /// <returns>A JSON result containing the predictions.</returns>
        [HttpGet("/Prediction/GetPredictions/{ticker}")]
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

        /// <summary>
        /// Clears all stored predictions from the database.
        /// </summary>
        /// <returns>A success message upon removing predictions.</returns>
        [HttpPost("/Prediction/ClearPredictions")]
        public IActionResult ClearPredictions()
        {
            _GetDataTools.ClearStockPredictions();
            return Ok("Predictions removed successfully.");
        }

        /// <summary>
        /// Adds a new set of predictions for a specific stock ticker to the database.
        /// </summary>
        /// <param name="ticker">The stock ticker symbol for which to add predictions.</param>
        /// <returns>A result message indicating the outcome of the addition.</returns>
        [HttpPost("/Prediction/AddPredictions/{ticker}")]
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

        //[HttpPost("/Prediction/AddPredictions")]
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
    }
}
