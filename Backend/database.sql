CREATE TABLE UserTypes (
    UserTypeId int AUTO_INCREMENT PRIMARY KEY,
    UserTypeName nvarchar(255)
);
INSERT INTO UserTypes VALUES ('Admin'), ('Client');

CREATE TABLE Users (
    UserId int AUTO_INCREMENT PRIMARY KEY,
    Email nvarchar(63) NOT NULL UNIQUE,
    Password nvarchar(63) NOT NULL,
    UserTypeId int,  
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE Users ADD CONSTRAINT FK_Users_UserTypeId_UserTypes_UserTypeId FOREIGN KEY (UserTypeId) REFERENCES UserTypes (UserTypeId);


CREATE TABLE Stocks (
    Ticker nvarchar(10) PRIMARY KEY,
    Name nvarchar(100) NOT NULL,
    OneDayPercentage DECIMAL(10,2),
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE UserStocks (
    UserId INT,
    Ticker nvarchar(10),
    Quantity DECIMAL(10, 6) NOT NULL DEFAULT 0.00, 
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT PK_StockPrice PRIMARY KEY CLUSTERED (Ticker, UserId)
);

ALTER TABLE UserStocks ADD CONSTRAINT FK_UserStocks_UserId_Users_UserId FOREIGN KEY (UserId) REFERENCES Users (UserId);
ALTER TABLE UserStocks ADD CONSTRAINT FK_UserStocks_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker);

CREATE TABLE News ( 
    NewsId INT AUTO_INCREMENT PRIMARY KEY,
    Title nvarchar(255) NOT NULL,
    Content TEXT NOT NULL,
    PublishedAt datetime NOT NULL
);

CREATE TABLE StockPrices (
    Ticker nvarchar(10) NOT NULL,
    Price DECIMAL(12,5) NOT NULL,
    Time datetime NOT NULL,
    CONSTRAINT PK_StockPrice PRIMARY KEY CLUSTERED (Ticker, Time)
);
ALTER TABLE StockPrices ADD CONSTRAINT FK_StockPrices_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker);

CREATE TABLE QuickStocks (
    Ticker nvarchar(10) PRIMARY KEY
);
ALTER TABLE QuickStocks ADD CONSTRAINT FK_QuickStocks_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker);


-- Inserts for dummies
INSERT INTO Users (email, password, CreatedAt) VALUES 
('mwwenger13@gmail.com', 'UIO*uio8', CURRENT_TIMESTAMP),
('infernothunder13@gmail.com', 'secure_password', CURRENT_TIMESTAMP),
('monkey12@gmail.com', 'password', CURRENT_TIMESTAMP);

INSERT INTO Stocks (name, ticker, CreatedAt) VALUES 
('Apple', 'AAPL', CURRENT_TIMESTAMP);

INSERT INTO UserStocks (user_id, stock_id, quantity, created_at) VALUES 
(1, 1, 100, CURRENT_TIMESTAMP);

INSERT INTO News (title, content, PublishedAt) VALUES 
('Moneky Big', 'Monkey Is Big', CURDATE());

INSERT INTO StockPrices (Ticker, Price, Time) VALUES 
('AAPL', 182.59, CURDATE());

INSERT INTO QuickStocks (Ticker) VALUES
    ('AAPL'),('GOOG'),('MSFT;')

