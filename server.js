
console.log('Server is starting...');

const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 3002;

app.use(cors()); // Enable CORS for all requests
app.use(bodyParser.json()); // Parse JSON bodies

// MySQL connection
const connection = mysql.createConnection({
  host: '127.0.0.1',
  user: 'root',
  password: 'DollarS1gn$',
  database: 'stock_trading_app'
});

connection.connect(error => {
  if (error) throw error;
  console.log('Successfully connected to the database.');
});

// GET request to fetch stocks
app.get('/stocks', (req, res) => {
  connection.query('SELECT * FROM stocks', (error, results) => {
    if (error) {
      return res.status(500).send(error);
    }
    res.status(200).json(results);
  });
});
//Fetch All Users
app.get('/users', (req, res) => {
    connection.query('SELECT * FROM users', (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      res.status(200).json(results);
    });
  });

  //Create a New User
  app.post('/users', (req, res) => {
    const { email, password } = req.body; // TODO: Ensure password is hashed for security
    const query = 'INSERT INTO users (email, password) VALUES (?, ?)';
    connection.query(query, [email, password], (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      res.status(201).json({ message: 'User created successfully', userId: results.insertId });
    });
  });

  //User Authentication
  app.post('/login', (req, res) => {
    const { email, password } = req.body;
    const query = 'SELECT * FROM users WHERE email = ?';
    connection.query(query, [email], (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      if (results.length > 0) {
        // TODO: Compare hashed password
        const user = results[0];
        if (password === user.password) { // Replace with hashed password comparison
          return res.status(200).json({ message: 'Login successful', userId: user.user_id });
        }
      }
      res.status(401).send('Invalid credentials');
    });
  });
  
//Fetch Stocks for a Specific User
app.get('/user/:userId/stocks', (req, res) => {
    const { userId } = req.params;
    const query = `
      SELECT stocks.*, user_stocks.quantity FROM user_stocks
      JOIN stocks ON user_stocks.stock_id = stocks.stock_id
      WHERE user_stocks.user_id = ?
    `;
    connection.query(query, [userId], (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      res.status(200).json(results);
    });
  });

  //Add a Stock to a User's Portfolio
app.post('/user/:userId/stocks', (req, res) => {
  const { userId } = req.params;
  const { stockId, quantity } = req.body;
  const query = 'INSERT INTO user_stocks (user_id, stock_id, quantity) VALUES (?, ?, ?)';
  connection.query(query, [userId, stockId, quantity], (error, results) => {
    if (error) {
      return res.status(500).send(error);
    }
    res.status(201).json({ message: 'Stock added to portfolio', userStockId: results.insertId });
  });
});

//Fetch Prices for a Specific Stock
app.get('/stocks/:stockId/prices', (req, res) => {
    const { stockId } = req.params;
    const query = 'SELECT * FROM stock_prices WHERE stock_id = ?';
    connection.query(query, [stockId], (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      res.status(200).json(results);
    });
  });

  //Fetch Recent News Articles
  app.get('/news', (req, res) => {
    connection.query('SELECT * FROM news', (error, results) => {
      if (error) {
        return res.status(500).send(error);
      }
      res.status(200).json(results);
    });
  });
  

  
// Start the Express server
app.listen(port, () => {
  console.log(`API server running on port ${port}`);
});

//simople tes
app.get('/test-db', (req, res) => {
    connection.query('SELECT 1 + 1 AS solution', (error, results) => {
      if (error) {
        return res.status(500).send('Database connection failed: ' + error);
      }
      res.status(200).send('Database connection successful. Answer: ' + results[0].solution);
    });
  });
  