import React, { useEffect, useState } from 'react';
import ErrorLog from "../Interfaces/ErrorLog";
import { getErrorLogs } from "../Services/ErrorLogService";

const ErrorLogsView: React.FC = () => {
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);

  useEffect(() => {
    const fetchErrorLogs = async () => {
      const logs = await getErrorLogs();
      setErrorLogs(logs);
    };

    fetchErrorLogs();
  }, []);

  return (
    <div>
      {errorLogs.map(log => (
        <div key={log.id}>
          <p><strong>Timestamp:</strong> {log.createdAt ? new Date(log.createdAt).toLocaleString() : 'N/A'}</p>
          <p><strong>Message:</strong> {log.message || 'No message'}</p>
          <hr />
        </div>
      ))}
    </div>
  );
};

export default ErrorLogsView;