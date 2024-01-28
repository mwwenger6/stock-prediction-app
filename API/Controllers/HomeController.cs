using Microsoft.AspNetCore.Mvc;
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

        [HttpGet("/Home/AddStockPrices/{ticker}/{interval}")]
        public IActionResult AddStockPrices(string ticker)
        {
            
        }





    }
}
