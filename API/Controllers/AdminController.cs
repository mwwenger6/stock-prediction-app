using Stock_Prediction_API.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MySqlConnector;
using Stock_Prediction_API.Entities;


namespace Stock_Prediction_API.Controllers
{
    public class AdminController : ControllerHelper
    {
        public AdminController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        [HttpGet("/Admin/GetErrorLogs")]
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

        [HttpPost("/Admin/DeleteErrorLog/{id}")]
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


        [HttpPost("/Admin/ChangeUserType/{email}/{userTypeName}")]
        public IActionResult ChangeUserType(string email, string userTypeName)
        {
            try
            {
                //Check that user type name is in db, get id of that type
                List<UserType> userTypes = _GetDataTools.GetUserTypes().ToList();
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

        [HttpPost("/Admin/AddMarketHolidays")]
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
    }

}