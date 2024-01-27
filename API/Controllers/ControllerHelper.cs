using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Stock_Prediction_API.Services;
using System.Runtime.CompilerServices;

#nullable disable
namespace Stock_Prediction_API.Controllers
{
    public class ControllerHelper : Controller
    {

        protected readonly AppDBContext _dbContext;
        //protected DbContextOptions<AppDBContext> _dbContextOptions;
        protected IConfiguration _configuration;
        protected dbTools _GetDataTools;

        public ControllerHelper(AppDBContext context, IConfiguration config) : base()
        {
            _dbContext = context;
            string activeConnectionString = config.GetValue<string>("ConnectionStrings:ActiveDBString");
            //_dbContextOptions = new DbContextOptionsBuilder<AppDBContext>()
            //    .UseSqlServer(config.GetConnectionString(activeConnectionString)).Options;
            _configuration = config;
            _GetDataTools = new(context);
        }
    }
}