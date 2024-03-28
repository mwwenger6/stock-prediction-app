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


INSERT INTO Stocks (Name, Ticker, CreatedAt) VALUES 
('Google', 'GOOG', CURRENT_TIMESTAMP),
('Microsoft', 'MSFT', CURRENT_TIMESTAMP);

INSERT INTO QuickStocks (Ticker) VALUES
    ('AAPL'),('GOOG'),('MSFT');

ALTER TABLE Stocks CHANGE COLUMN OneDayPercentage CurrentPrice DECIMAL(10,2);

CREATE TABLE ErrorLogs (
    ErrorId INT AUTO_INCREMENT PRIMARY KEY,
    ErrorMessage VARCHAR(255),
    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE Stocks 
ADD COLUMN updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
ADD COLUMN dailyChange FLOAT;

 Update Users Set UserTypeId = 1 Where UserTypeId IS NULL;

 CREATE TABLE UserWatchlistStocks (
    WatchlistId INT AUTO_INCREMENT PRIMARY KEY,
    UserId INT,
    Ticker NVARCHAR(10),
    CONSTRAINT FK_UserWatchlistStocks_UserId FOREIGN KEY (UserId) REFERENCES Users (UserId),
    CONSTRAINT FK_UserWatchlistStocks_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker)
);

CREATE TABLE StockPredictions (
    Ticker NVARCHAR(10) NOT NULL,
    PredictedPrice DECIMAL(12,2) NOT NULL,
    PredictionOrder INT NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT PK_StockPredictions PRIMARY KEY (Ticker, PredictedPrice, PredictionOrder),
    CONSTRAINT FK_StockPredictions_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker)
);

CREATE TABLE MarketHolidays (
    Day DATE PRIMARY KEY
);

CREATE TABLE StockPrices_5Min (
    Ticker nvarchar(10) NOT NULL,
    Price DECIMAL(12,5) NOT NULL,
    Time datetime(3) NOT NULL, -- datetime(3) for millisecond precision, important for 5-minute intervals
    CONSTRAINT PK_StockPrices_5Min PRIMARY KEY CLUSTERED (Ticker, Time)
);
ALTER TABLE StockPrices_5Min ADD CONSTRAINT FK_StockPrices_5Min_Ticker_Stocks_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker);

CREATE TABLE SupportedStocks (
    Ticker nvarchar(10) PRIMARY KEY,
    Name nvarchar(100) NOT NULL,
    LastUpdated datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    IsActive BOOLEAN DEFAULT TRUE
);

alter table StockPredictions drop column CreatedAt;
ALTER TABLE Users ADD UserVerified BIT NOT NULL DEFAULT 0;
ALTER TABLE Users ADD VerificationCode NVARCHAR(12);

CREATE TABLE HistoricalStockPredictions (
    PredictionId INT AUTO_INCREMENT PRIMARY KEY,
    Ticker NVARCHAR(10) NOT NULL,
    PredictedPrice DECIMAL(12,2) NOT NULL,
    PredictionDate DATE NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT FK_HistoricalStockPredictions_Ticker FOREIGN KEY (Ticker) REFERENCES Stocks (Ticker)
);

ALTER TABLE SupportedStocks
MODIFY COLUMN Name NVARCHAR(200) NOT NULL;

ALTER TABLE StockPrices
ADD COLUMN IsClosePrice BIT NOT NULL DEFAULT 0;

UPDATE StockPrices
SET IsClosePrice = 1;

SELECT MIN(Time) AS OldestTime FROM StockPrices_5Min;

DELETE FROM StockPrices WHERE Time >= '2024-03-01 09:30:00';

INSERT INTO StockPrices (Ticker, Price, Time, IsClosePrice)
SELECT Ticker, Price, Time, 0 -- Initially mark all as not close prices
FROM StockPrices_5Min;

UPDATE StockPrices
SET IsClosePrice = 1
WHERE TIME(Time) BETWEEN '15:55:00' AND '15:59:59';

ALTER TABLE News
ADD COLUMN PhotoUrl NVARCHAR(255),
ADD COLUMN ArticleUrl NVARCHAR(255),
ADD COLUMN Category NVARCHAR(50),
ADD COLUMN Summary TEXT;


ALTER TABLE News
ADD COLUMN RelatedStockTicker NVARCHAR(10);


ALTER TABLE News
ADD CONSTRAINT FK_News_RelatedStockTicker_Stocks_Ticker
FOREIGN KEY (RelatedStockTicker) REFERENCES Stocks (Ticker);

UPDATE Stocks
SET Name = 'Advanced Micro Devices'
WHERE Ticker = 'AMD';

DROP TABLE StockPrices_5Min;


ALTER TABLE UserStocks
ADD COLUMN Price DECIMAL(12, 2) NOT NULL DEFAULT 0.00;
