// db.js
const mysql = require('mysql2/promise');

const pool = mysql.createPool({
  host: '71.113.172.111',
  user: 'appuser',
  password: 'secure_password',
  database: 'stock_trading_app',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// Helper function to execute SQL queries
const query = async (sql, params) => {
  const [rows, fields] = await pool.execute(sql, params);
  return rows;
};

module.exports = {
  query,
  pool
};
