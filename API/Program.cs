using CNSPortal_API.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Stock_Prediction_API.Services;
using Microsoft.OpenApi.Models;
using Stock_Prediction_API.Controllers; // Add this namespace for OpenApiInfo

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
string activeConnectionString = builder.Configuration.GetValue<string>("ConnectionStrings:ActiveDBString");
string connectionString = builder.Configuration.GetConnectionString(activeConnectionString);

builder.Services.AddDbContext<AppDBContext>(options =>
    options.UseMySql(builder.Configuration.GetConnectionString("PredictionAPI"),
    new MySqlServerVersion(new Version(8, 3, 0))));

builder.Services.AddControllers();

//builder.Services.AddScoped<HomeController>();
//builder.Services.AddHostedService<MyBackgroundService>();

builder.Services.AddDistributedMemoryCache();

// Register authorization services
builder.Services.AddAuthorization();

// If your application uses authentication, make sure to add authentication services as well
builder.Services.AddAuthentication();

// Register Swagger services
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "Your API", Version = "v1" });
});

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAnyOrigin",
        builder =>
        {
            builder.AllowAnyOrigin()
                   .AllowAnyMethod()
                   .AllowAnyHeader();
        });
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

// Middleware
app.UseMiddleware<ApiKeyMiddleware>();

// Make sure to call UseAuthentication before UseAuthorization if you're using both
app.UseAuthentication();
app.UseAuthorization();

app.UseCors("AllowAnyOrigin");

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers(); // This will map attribute routes
});


await app.RunAsync();
