CREATE TABLE [dbo].[UserTypes] (
    [UserTypeId] int IDENTITY(1,1) PRIMARY KEY,
    [UserTypeName] nvarchar(255)
)
INSERT INTO dbo.UserTypes VALUES ('Admin'), ('Client');

CREATE TABLE [dbo].[Users] (
    [UserId] int IDENTITY(1,1) PRIMARY KEY,
    [Email] nvarchar(63) NOT NULL UNIQUE,
    [Password] nvarchar(63) NOT NULL,
    [UserTypeId] int,  --FOREIGN KEY
    [CreatedAt] TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

ALTER TABLE [dbo].[Users] ADD CONSTRAINT FK_Users_UserTypeId_UserTypes_UserTypeId FOREIGN KEY (UserTypeId) REFERENCES [dbo].[UserTypes] (UserTypeId)


CREATE TABLE [dbo].[Stocks] (
    [Ticker] nvarchar(10) PRIMARY KEY,
    [Name] nvarchar(100) NOT NULL,
    [OneDayPercentage] DECIMAL(10,2),
    [CreatedAt] TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE [dbo].[UserStocks] (
    [UserId] INT,
    [Ticker] nvarchar(10),
    [Quantity] DECIMAL(10, 6) NOT NULL DEFAULT 0.00, -- track quanitity of stocks
    [CreatedAt] TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT PK_StockPrice PRIMARY KEY CLUSTERED (Ticker, UserId)
);

ALTER TABLE [dbo].[UserStocks] ADD 
    CONSTRAINT FK_UserStocks_UserId_Users_UserId FOREIGN KEY (UserId) REFERENCES [dbo].[Users] (UserId)
    CONSTRAINT FK_UserStocks_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES [dbo].[Stocks] (Ticker)

CREATE TABLE [dbo].[News] ( --not sure about storing articles, and what data specifically, subject to change
    [NewsId] INT IDENTITY(1,1) PRIMARY KEY,
    [Title] nvarchar(255) NOT NULL,
    [Content] TEXT NOT NULL,
    [PublishedAt] datetime NOT NULL,
);

CREATE TABLE [dbo].[StockPrices] (
    [Ticker] nvarchar(10) NOT NULL,
    [Price] DECIMAL(8,6) NOT NULL,
    [Time] datetime NOT NULL,
    CONSTRAINT PK_StockPrice PRIMARY KEY CLUSTERED (Ticker, Time)
)

ALTER TABLE [dbo].[StockPrices] ADD 
    CONSTRAINT FK_StockPrices_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES [dbo].[Stocks] (Ticker)

INSERT INTO [dbo].[Users] VALUES ('mwwenger13@gmail.com', 'UIO*uio8', 1), ('infernothunder13@gmail.com', 'secure_password', 1), ('monkey12@gmail.com', 'password', 2)
INSERT INTO [dbo].[Stocks] VALUES ('AAPL', 'Apple', -0.85)
INSERT INTO [dbo].[UserStocks] VALUES (1, 'secure', 2)
INSERT INTO [dbo].[News] VALUES ('Moneky Big', 'Monkey Is Big', GETDATE())
INSERT INTO [dbo].[StockPrices] VALUES ('AAPL', 182.59, GETDATE())