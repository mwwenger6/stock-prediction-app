const express = require('express');
const router = express.Router();
const db = require('./db');

// Get all users
router.get('/', async (req, res) => {
    try {
        const users = await db.query('SELECT * FROM users');
        res.json(users);
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).send('Error fetching users');
    }
});

// Get single user
router.get('/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const user = await db.query('SELECT * FROM users WHERE user_id = ?', [userId]);
        res.json(user);
    } catch (error) {
        console.error('Error fetching user:', error);
        res.status(500).send('Error fetching user');
    }
});

// Create new user
router.post('/', async (req, res) => {
    try {
        const { email, password } = req.body;
        const result = await db.query('INSERT INTO users (email, password) VALUES (?, ?)', [email, password]);
        res.status(201).json({ message: 'User created successfully', userId: result.insertId });
    } catch (error) {
        console.error('Error creating user:', error);
        res.status(500).send('Error creating user');
    }
});

// Update user
router.put('/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const { email, newPassword } = req.body;
        await db.query('UPDATE users SET email = ?, password = ? WHERE user_id = ?', [email, newPassword, userId]);
        res.json({ message: 'User updated successfully' });
    } catch (error) {
        console.error('Error updating user:', error);
        res.status(500).send('Error updating user');
    }
});

// Delete user
router.delete('/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        await db.query('DELETE FROM users WHERE user_id = ?', [userId]);
        res.json({ message: 'User deleted successfully' });
    } catch (error) {
        console.error('Error deleting user:', error);
        res.status(500).send('Error deleting user');
    }
});

module.exports = router;
