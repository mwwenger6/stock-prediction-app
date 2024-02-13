CREATE TABLE StockData (
    StockID int not null,
    StockDate datetime not null,
    StockClosePrice float,
    PRIMARY KEY (StockID, StockDate)
);

CREATE TABLE Stocks (
    StockID INT IDENTITY(1,1) PRIMARY KEY,
    StockName VARCHAR(100) not null,
    StockSymbol VARCHAR(10) not null,
);

ALTER TABLE StockData ADD CONSTRAINT FK_StockData_StockID_Stocks_StockID
FOREIGN KEY (StockID) REFERENCES Stocks(StockID);

SELECT * FROM StockData;
SELECT * FROM Stocks;

--View how much space is being used
SELECT
    DB_NAME(database_id) AS 'Database Name',
    name AS 'Logical Name',
    physical_name AS 'Physical Path',
    (size * 8) / 1024 AS 'Total Size (MB)',
    (fileproperty(name, 'SpaceUsed') * 8) / 1024 AS 'Used Space (MB)',
    ((size - fileproperty(name, 'SpaceUsed')) * 8) / 1024 AS 'Free Space (MB)'
FROM
    sys.master_files
WHERE
    database_id = DB_ID('master'); --master is db name