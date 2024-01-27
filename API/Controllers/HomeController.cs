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
    }
}
