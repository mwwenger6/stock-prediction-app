using Stock_Prediction_API.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using MySqlConnector;
using Stock_Prediction_API.Entities;


namespace Stock_Prediction_API.Controllers
{
    public class UserController : ControllerHelper
    {
        public UserController(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base(context, config, web) { }

        [HttpGet("/User/GetUsers")]
        public IActionResult GetUsers()
        {
            try
            {
                List<User> users = _GetDataTools.GetUsers().ToList();
                foreach (User user in users)
                {
                    if (user.TypeId == 1)
                        user.TypeName = "Admin";
                    else
                        user.TypeName = "Client";
                }
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

        [HttpGet("/User/GetUser/{email}")]
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

        [HttpGet("/User/AuthenticateUser/{email}/{password}")]
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

        [HttpGet("/User/GetUserStocks/{userId}")]
        public IActionResult GetUserStocks(int userId)
        {
            try
            {
                return Json(_GetDataTools.GetUserStocks(userId).ToList());
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
        }

        [HttpGet("/User/GetUserStock/{userId}/{ticker}")]
        public IActionResult GetUserStock(int userId, string ticker)
        {
            try
            {
                return Json(_GetDataTools.GetUserStock(userId, ticker));
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

        [HttpGet("/User/GetUserStockData/{userId}")]
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
            foreach (UserStock userStock in userStocks)
            {

                List<StockPrice> prices = new();
                try
                {
                    prices = _GetDataTools.GetStockPrices(userStock.Ticker, GetEasternTime().Date).ToList();
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
                foreach (StockPrice stockPrice in prices)
                {
                    userStockPrices[count] = userStockPrices[count] + (stockPrice.Price * userStock.Quantity);
                    count++;
                }
            }
            return Json(userStockPrices);
        }

        [HttpPost("/User/DeleteUser/{email}")]
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
        [HttpPost("/User/AddUser")]
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

        [HttpPost("/User/VerifyUser/{code}")]
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
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Error With Verifying User. {ex.Message}");
            }
        }

        [HttpPost("/User/AddUserStock/{userId}/{ticker}/{quantity}/{price}")]
        public IActionResult AddUserStock(int userId, string ticker, float quantity, float price)
        {
            try
            {
                UserStock? userStock = _GetDataTools.GetUserStockNullable(userId, ticker);
                if (userStock != null)
                {
                    price = ((userStock.Price * userStock.Quantity) + (price * quantity))/(userStock.Quantity + quantity) ;
                    quantity = userStock.Quantity + quantity;
                }
                _GetDataTools.AddUserStock(new UserStock
                {
                    UserId = userId,
                    Ticker = ticker,
                    Quantity = quantity,
                    Price = price,
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

        [HttpPost("/User/SubtractUserStock/{userId}/{ticker}/{quantity}")]
        public IActionResult SubtractUserStock(int userId, string ticker, float quantity)
        {
            try
            {
                UserStock? userStock = _GetDataTools.GetUserStockNullable(userId, ticker) ?? throw new Exception("No UserStock found");
                float newQuantity = userStock.Quantity - quantity;
                if (newQuantity < 0)
                    throw new Exception("Cannot sell more shares than you own");
                _GetDataTools.AddUserStock(new UserStock
                {
                    UserId = userId,
                    Ticker = ticker,
                    Quantity = newQuantity,
                    Price = userStock.Price
                });
            }
            catch (Exception ex)
            {
                _GetDataTools.LogError(new()
                {
                    Message = ex.Message,
                    CreatedAt = GetEasternTime(),
                });
                return StatusCode(500, $"Could not subtract Stock for User: {userId}. {ex.Message}");
            }
            return Ok("Stock Added Successfully.");
        }

        [HttpPost("/User/RemoveUserStock/{userId}/{ticker}")]
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
    }
}
