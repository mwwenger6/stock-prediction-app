//var builder = WebApplication.CreateBuilder(args);

//// Add services to the container.
//string activeConnectionString = builder.Configuration.GetValue<string>("ConnectionStrings:ActiveDBString");

//string connectionString = builder.Configuration.GetConnectionString(activeConnectionString);
////builder.Services.add;
//builder.Services.AddDbContext<PortalDbContext>(options =>
//    options.UseSqlServer(connectionString));
//builder.Services.AddDistributedMemoryCache();

//var app = builder.Build();

//// Configure the HTTP request pipeline.
//if (app.Environment.IsDevelopment())
//{
//    app.UseSwagger();
//    app.UseSwaggerUI();
//}

//app.UseHttpsRedirection();

//app.UseAuthorization();

//app.UseEndpoints(endpoints =>
//{
//    endpoints.MapControllers(); // This will map attribute routes
//});

//app.Run();
