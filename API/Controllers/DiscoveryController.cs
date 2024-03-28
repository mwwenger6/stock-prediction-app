using Stock_Prediction_API.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MySqlConnector;
using Stock_Prediction_API.Entities;

/// <summary>
/// Handles administrative actions such as managing error logs, user types, and market holidays.
/// Inherits from ControllerHelper to utilize shared functionality.
/// </summary>  

namespace Stock_Prediction_API.Controllers
{
    public class DiscoveryController : ControllerHelper

    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AdminController"/> class.
        /// </summary>
        /// <param name="context">The database context for accessing the database.</param>
        /// <param name="config">The configuration properties, used to access configuration settings like connection strings.</param>
        /// <param name="web">The web host environment, providing information about the web hosting environment an application is running in.</param>

        public DiscoveryController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        [HttpGet("/Discovery/GetBiggestGainers")]
        public IActionResult GetBiggestGainers()
        {
            try
            {
                return Json(_GetDataTools.GetVolatileStocks(true));
            }
            catch (Exception ex)
            {
                //_GetDataTools.LogError(new()
                //{
                //    Message = ex.Message,
                //    CreatedAt = GetEasternTime(),
                //});
                //return StatusCode(500, $"Could not get user stock: {userId}. {ex.Message}");
                return Json(null);
            }
        }

        [HttpGet("/Discovery/GetBiggestLosers")]
        public IActionResult GetBiggestLosers()
        {
            try
            {
                List<VolatileStock> stocks = _GetDataTools.GetVolatileStocks(false).ToList();
                return Json(stocks);
            }
            catch (Exception ex)
            {
                //_GetDataTools.LogError(new()
                //{
                //    Message = ex.Message,
                //    CreatedAt = GetEasternTime(),
                //});
                //return StatusCode(500, $"Could not get user stock: {userId}. {ex.Message}");
                return Json(null);
            }
        }





        [HttpPost("/Discovery/AddVolatileStocks")]
        public IActionResult AddVolatileStocks([FromBody] List<VolatileStock> stocks)
        {
            if(stocks != null)
            {
                try
                {
                    _GetDataTools.AddVolatileStocks(stocks);
                }
                catch(Exception ex)
                {
                    _GetDataTools.LogError(new()
                    {
                        Message = ex.Message,
                        CreatedAt = GetEasternTime(),
                    });
                    return StatusCode(500, $"Could not add VolatileStocks. {ex.Message}");
                }
            }
            return Ok("Volatile Stocks Added");
        }

    }

}