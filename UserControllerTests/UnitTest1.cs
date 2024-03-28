using Xunit;
using Moq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.AspNetCore.Hosting;
using Stock_Prediction_API.Controllers;
using Stock_Prediction_API.Entities;
using Stock_Prediction_API.Services;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;

public class UserControllerTests
{
    private readonly UserController _controller;
    private readonly Mock<AppDBContext> _mockContext = new Mock<AppDBContext>();
    private readonly Mock<IConfiguration> _mockConfiguration = new Mock<IConfiguration>();
    private readonly Mock<IWebHostEnvironment> _mockWebHostEnvironment = new Mock<IWebHostEnvironment>();


    private readonly Mock<dbTools> _mockDbTools = new Mock<dbTools>();
    public UserControllerTests()
    {
        // Mock dbTools or any other service that UserController depends on
        _mockDbTools.Setup(m => m.GetUsers()).ReturnsAsync(new List<User> { /* user data */ });

        // Mock IConfiguration and IWebHostEnvironment as necessary

        // Instantiate UserController with mocks
        _controller = new UserController(_mockDbTools.Object, _mockConfiguration.Object, _mockWebHostEnvironment.Object);
    }


    [Fact]
    public void GetUsers_ReturnsJsonResult_WithListOfUsers()
    {
        // Act
        var result = _controller.GetUsers();

        // Assert
        var jsonResult = Assert.IsType<JsonResult>(result);
        var users = Assert.IsAssignableFrom<IEnumerable<User>>(jsonResult.Value);
        Assert.NotEmpty(users);
    }

    [Fact]
    public void GetUser_ReturnsJsonResult_WithUser()
    {
        // Arrange
        var email = "test@example.com";
        _mockDbTools.Setup(m => m.GetUser(email)).Returns(new User { /* Initialized with email */ });

        // Act
        var result = _controller.GetUser(email);

        // Assert
        var jsonResult = Assert.IsType<JsonResult>(result);
        var user = Assert.IsType<User>(jsonResult.Value);
        Assert.Equal(email, user.Email); // Assuming User entity has an Email property
    }
}
