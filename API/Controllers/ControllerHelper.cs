using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Stock_Prediction_API.Services;
using Stock_Prediction_API.Services.API_Tools;
using System.Runtime.CompilerServices;

#nullable disable
namespace Stock_Prediction_API.Controllers
{
    /// <summary>
    /// Serves as a base controller class providing shared resources and functionality
    /// for other controllers in the Stock Prediction API.
    /// </summary>
    public class ControllerHelper : Controller
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="ControllerHelper"/> class.
        /// </summary>
        /// <param name="context">The database context for accessing the database.</param>
        /// <param name="config">The configuration properties, used to access configuration settings like connection strings.</param>
        /// <param name="web">The web host environment, providing information about the web hosting environment an application is running in.</param>
        
        protected readonly AppDBContext _dbContext;
        protected DbContextOptions<AppDBContext> _dbContextOptions;
        protected IConfiguration _configuration;
        protected dbTools _GetDataTools;
        public readonly TwelveDataTools _TwelveDataTools;
        public readonly FinnhubAPITools _FinnhubDataTools;
        public readonly EmailTools _EmailTools;
        public ControllerHelper(AppDBContext context, IConfiguration config, IWebHostEnvironment web) : base()
        {
            _dbContext = context;
            string activeConnectionString = config.GetValue<string>("ConnectionStrings:ActiveDBString");
            _dbContextOptions = new DbContextOptionsBuilder<AppDBContext>()
                .UseMySql(config.GetConnectionString(activeConnectionString), new MySqlServerVersion(new Version(8, 3, 0)))
                .Options;
            _configuration = config;
            _GetDataTools = new dbTools(context, _dbContextOptions);
            _FinnhubDataTools = new(config);        
            _TwelveDataTools = new(config);
            _EmailTools = new(config, web);
        }

        /// <summary>
        /// Gets the current time in Eastern Standard Time (EST).
        /// </summary>
        /// <returns>The current time converted to Eastern Standard Time.</returns>

        public DateTime GetEasternTime()
        {
            DateTime currentTime = DateTime.UtcNow;
            return TimeZoneInfo.ConvertTimeBySystemTimeZoneId(currentTime, TimeZoneInfo.Utc.Id, "Eastern Standard Time");
        }

    }
}