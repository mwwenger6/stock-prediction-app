import React, { useEffect, useState } from 'react';
import ErrorLog from "../Interfaces/ErrorLog";
import User from "../Interfaces/User"; // Import the User interface
import { getErrorLogs, getUsers } from "../Services/ErrorLogService"; // Ensure getUsers is imported

const ErrorLogsView: React.FC = () => {
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [users, setUsers] = useState<User[]>([]); // Add state for users

  useEffect(() => {
    const fetchData = async () => {
      const logs = await getErrorLogs();
      setErrorLogs(logs);
      const users = await getUsers(); // Fetch users
      setUsers(users);
    };

    fetchData();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}> {/* Adjust layout as needed */}
      <div>
        {errorLogs.map(log => (
          <div key={log.id}>
            <p><strong>Timestamp:</strong> {log.createdAt ? new Date(log.createdAt).toLocaleString() : 'N/A'}</p>
            <p><strong>Message:</strong> {log.message || 'No message'}</p>
            <hr />
          </div>
        ))}
      </div>
      <div> {/* New table for users */}
        <table>
          <thead>
            <tr>
              <th>UserId</th>
              <th>Email</th>
              <th>UserTypeId</th> {/* Adjust according to how you want to display user types */}
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.id}>
                <td>{user.id}</td>
                <td>{user.email}</td>
                <td>{user.typeId}</td> {/* Assuming typeId is available, adjust as needed */}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ErrorLogsView;
