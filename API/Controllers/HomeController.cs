using Microsoft.AspNetCore.Mvc;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;

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
                List<User> user = _GetDataTools.GetUsers().ToList();
                int id = user.First().Id;
            }
            catch(Exception ex)
            {

            }
            return View();
        }


        [HttpGet("/Home/GetRecentStockPrice")]
        public IActionResult GetRecentStockPrice(string ticker)
        {
            try
            {
                var recentPrice = _GetDataTools.GetRecentStockPrice(ticker);
                if (recentPrice == null)
                {
                    return NotFound("Stock price not found.");
                }

                return View(recentPrice);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }


        [HttpGet("/Home/GetStockPrices")]
        public IActionResult GetStockPrices(string ticker, int interval)
        {
            try
            {
                var stockPrices = _GetDataTools.GetStockPrices(ticker, interval).ToList();
                if (stockPrices == null || !stockPrices.Any())
                {
                    return NotFound("Stock prices not found.");
                }

                return View(stockPrices);
            }
            catch (Exception ex)
            {
                // Log the exception
                return StatusCode(500, "Internal server error");
            }
        }







    }
}
