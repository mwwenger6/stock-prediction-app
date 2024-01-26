const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const stockRoutes = require('./stockRoutes');
const userRoutes = require('./userRoutes');
const newsRoutes = require('./newsRoutes');

const app = express();
const port = process.env.PORT || 3002; // Use environment variable for port, default to 3002

// CORS setup
const corsOptions = {
  origin: '*', // For development, you might use '*'. For production, specify the actual origin.
  optionsSuccessStatus: 200,
};
app.use(cors(corsOptions));

// Body parser middleware
app.use(bodyParser.json());

// Use routes
app.use('/stocks', stockRoutes);
app.use('/users', userRoutes);
app.use('/news', newsRoutes);

// Start the server
app.listen(port, () => {
  console.log(`API server running on port ${port}`);
});
