using CNSPortal_API.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Stock_Prediction_API.Services;

var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    WebRootPath = "wwwroot",
    ContentRootPath = Environment.CurrentDirectory
});

// Add services to the container.
string activeConnectionString = builder.Configuration.GetValue<string>("ConnectionStrings:ActiveDBString");
string connectionString = builder.Configuration.GetConnectionString(activeConnectionString);

builder.Services.AddDbContext<AppDBContext>(options =>
    options.UseMySql(builder.Configuration.GetConnectionString("PredictionAPI"),
    new MySqlServerVersion(new Version(8, 3, 0))));

builder.Services.AddDistributedMemoryCache();

// Register authorization services
builder.Services.AddAuthorization();

// If your application uses authentication, make sure to add authentication services as well
builder.Services.AddAuthentication();

builder.Services.AddControllers();

// Configure Kestrel server to listen on port 80
builder.WebHost.ConfigureKestrel(serverOptions =>
{
    serverOptions.ListenAnyIP(80); // Listen for HTTP traffic on port 80
});

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
