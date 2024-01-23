
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
  host: '71.113.172.111',
  user: 'appuser',
  password: 'secure_password',   
  database: 'stock_trading_app'
});


connection.connect(error => {
  if (error) {
    console.error('Failed to connect to the database:', error);
    return;
  }
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

  //Read (Single User)
  app.get('/users/:userId', (req, res) => {
    const { userId } = req.params;
    const query = 'SELECT * FROM users WHERE user_id = ?';
    connection.query(query, [userId], (error, results) => {
        if (error) {
            return res.status(500).send(error);
        }
        res.status(200).json(results[0]);
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

//Update user
app.put('/users/:userId', (req, res) => {
  const { userId } = req.params;
  const { email, newPassword } = req.body; // Assume password is hashed
  const query = 'UPDATE users SET email = ?, password = ? WHERE user_id = ?';
  connection.query(query, [email, newPassword, userId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'User updated successfully' });
  });
});


//Delte user
app.delete('/users/:userId', (req, res) => {
  const { userId } = req.params;
  const query = 'DELETE FROM users WHERE user_id = ?';
  connection.query(query, [userId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'User deleted successfully' });
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

//create stock
app.post('/stocks', (req, res) => {
  const { name, ticker } = req.body;
  const query = 'INSERT INTO stocks (name, ticker) VALUES (?, ?)';
  connection.query(query, [name, ticker], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(201).json({ message: 'Stock added successfully', stockId: results.insertId });
  });
});

//update stock
app.put('/stocks/:stockId', (req, res) => {
  const { stockId } = req.params;
  const { name, newTicker } = req.body;
  const query = 'UPDATE stocks SET name = ?, ticker = ? WHERE stock_id = ?';
  connection.query(query, [name, newTicker, stockId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'Stock updated successfully' });
  });
});

//delete stock
app.delete('/stocks/:stockId', (req, res) => {
  const { stockId } = req.params;
  const query = 'DELETE FROM stocks WHERE stock_id = ?';
  connection.query(query, [stockId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'Stock deleted successfully' });
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

//Update user stock quantity
app.put('/user/:userId/stocks/:stockId', (req, res) => {
  const { userId, stockId } = req.params;
  const { quantity } = req.body;
  const query = 'UPDATE user_stocks SET quantity = ? WHERE user_id = ? AND stock_id = ?';
  connection.query(query, [quantity, userId, stockId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'User stock quantity updated successfully' });
  });
});

//delete a users stock
app.delete('/user/:userId/stocks/:stockId', (req, res) => {
  const { userId, stockId } = req.params;
  const query = 'DELETE FROM user_stocks WHERE user_id = ? AND stock_id = ?';
  connection.query(query, [userId, stockId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'User stock deleted successfully' });
  });
});

//creaet stock price
app.put('/stocks/:stockId/prices/:priceId', (req, res) => {
  const { stockId, priceId } = req.params;
  const { price } = req.body;
  const query = 'UPDATE stock_prices SET price = ? WHERE stock_id = ? AND stock_price_id = ?';
  connection.query(query, [price, stockId, priceId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'Stock price updated successfully' });
  });
});

//update stock price
app.put('/stocks/:stockId/prices/:priceId', (req, res) => {
  const { stockId, priceId } = req.params;
  const { price } = req.body;
  const query = 'UPDATE stock_prices SET price = ? WHERE stock_id = ? AND stock_price_id = ?';
  connection.query(query, [price, stockId, priceId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'Stock price updated successfully' });
  });
});

//delete the stock price
app.delete('/stocks/:stockId/prices/:priceId', (req, res) => {
  const { stockId, priceId } = req.params;
  const query = 'DELETE FROM stock_prices WHERE stock_id = ? AND stock_price_id = ?';
  connection.query(query, [stockId, priceId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'Stock price deleted successfully' });
  });
});

//create news article
app.post('/news', (req, res) => {
  const { title, content } = req.body;
  const query = 'INSERT INTO news (title, content) VALUES (?, ?)';
  connection.query(query, [title, content], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(201).json({ message: 'News article added successfully', newsId: results.insertId });
  });
});

//update news articl
app.put('/news/:newsId', (req, res) => {
  const { newsId } = req.params;
  const { title, content } = req.body;
  const query = 'UPDATE news SET title = ?, content = ? WHERE news_id = ?';
  connection.query(query, [title, content, newsId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'News article updated successfully' });
  });
});

//delete news article
app.delete('/news/:newsId', (req, res) => {
  const { newsId } = req.params;
  const query = 'DELETE FROM news WHERE news_id = ?';
  connection.query(query, [newsId], (error, results) => {
      if (error) {
          return res.status(500).send(error);
      }
      res.status(200).json({ message: 'News article deleted successfully' });
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

/*simople testinf method
app.get('/test-db', (req, res) => {
    connection.query('SELECT 1 + 1 AS solution', (error, results) => {
      if (error) {
        return res.status(500).send('Database connection failed: ' + error);
      }
      res.status(200).send('Database connection successful. Answer: ' + results[0].solution);
    });
  });
  */