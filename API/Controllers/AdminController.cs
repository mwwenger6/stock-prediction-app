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
    public class AdminController : ControllerHelper

    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AdminController"/> class.
        /// </summary>
        /// <param name="context">The database context for accessing the database.</param>
        /// <param name="config">The configuration properties, used to access configuration settings like connection strings.</param>
        /// <param name="web">The web host environment, providing information about the web hosting environment an application is running in.</param>
      
        public AdminController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        /// <summary>
        /// Retrieves all error logs from the database.
        /// </summary>
        /// <returns>A JSON representation of all error logs.</returns>

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

        /// <summary>
        /// Deletes a specific error log identified by its ID.
        /// </summary>
        /// <param name="id">The ID of the error log to delete.</param>
        /// <returns>A status indicating success or failure of the deletion.</returns>
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

        /// <summary>
        /// Changes the type of a user based on their email address.
        /// </summary>
        /// <param name="email">The email of the user to update.</param>
        /// <param name="userTypeName">The new user type name to assign.</param>
        /// <returns>A status message indicating the result of the operation.</returns>
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

        /// <summary>
        /// Adds market holidays retrieved from an external service to the database.
        /// </summary>
        /// <returns>A status message indicating the number of holidays added or an error message.</returns>
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