using CNSPortal_API.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Stock_Prediction_API.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
string activeConnectionString = builder.Configuration.GetValue<string>("ConnectionStrings:ActiveDBString");
string connectionString = builder.Configuration.GetConnectionString(activeConnectionString);

builder.Services.AddDbContext<AppDBContext>(options =>
    options.UseMySql(builder.Configuration.GetConnectionString("PredictionAPI"),
    new MySqlServerVersion(new Version(8, 3, 0))));
    mySqlOptions => mySqlOptions.EnableRetryOnFailure(
            // Maximum number of retry attempts
            maxRetryCount: 5,
            // Maximum delay between retries
            maxRetryDelay: TimeSpan.FromSeconds(10),
            // Specific MySQL error numbers to consider as transient
            errorNumbersToAdd: null)));

builder.Services.AddDistributedMemoryCache();

// Register authorization services
builder.Services.AddAuthorization();

// If your application uses authentication, make sure to add authentication services as well
builder.Services.AddAuthentication();

builder.Services.AddControllers();

var app = builder.Build();
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

// Middleware
app.UseMiddleware<ApiKeyMiddleware>();

// Make sure to call UseAuthentication before UseAuthorization if you're using both
app.UseAuthentication();
app.UseAuthorization();

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers(); // This will map attribute routes
});

app.Run();
