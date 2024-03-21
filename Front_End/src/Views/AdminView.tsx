import React, { useEffect, useState } from 'react';
import ErrorLog from "../Interfaces/ErrorLog";
import User from "../Interfaces/User";
import { getErrorLogs, getUsers } from "../Services/ErrorLogService";
import DataTable from 'react-data-table-component';

const AdminView: React.FC = () => {
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [users, setUsers] = useState<User[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      let logs = await getErrorLogs();
      logs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
      setErrorLogs(logs);
      const users = await getUsers();
      setUsers(users);
    };
    fetchData();
  }, []);

  const removeErrorLog = async (id:number) => {
    try {
      // Note the change to method: 'POST'
      const response = await fetch(`/Home/DeleteErrorLog/${id}`, { method: 'POST' });
      if (!response.ok) {
        throw new Error('Failed to delete the error log.');
      }
      // If deletion was successful, update the state to remove the log from the table
      setErrorLogs(prevLogs => prevLogs.filter(log => log.id !== id));
    } catch (error) {
      console.error('Error deleting error log:', error);
      // Optionally, display an error message to the user
    }
};

  
  


  const errorHeaders = [
      {
        name: 'Id',
        selector: (row: ErrorLog) => row.id,
        sortable: true,
        width: "100px",
      },
      {
        name: 'Actions',
        cell: (row: ErrorLog) => <button onClick={() => removeErrorLog(row.id)}>Remove</button>,
        ignoreRowClick: true,
      },
      {
        name: 'Message',
        selector: (row: ErrorLog) => row.message,
        sortable: false,
      },
      {
        name: 'Date',
        selector: (row: ErrorLog) => row.createdAt,
        sortable: true,
        width: "200px",
      },
  ];
    const userHeaders = [
        {
            name: 'User Id',
            selector: (row: User) => row.id,
            sortable: true,
            width: "100px",
        },
        {
            name: 'Email',
            selector: (row: User) => row.email,
            sortable: false,
        },
        {
          name: 'Is Verified',
          selector: (row: User) => row.isVerified,
          sortable: false,
        },
        {
            name: 'Account Type Id',
            selector: (row: User) => row.typeId,
            sortable: true,
            width: "150px",
        },
        {
            name: 'Creation Date',
            selector: (row: User) => row.createdAt,
            sortable: true,
            width: "200px",
        },
    ];

  return (
    <div className={"m-md-4 m-2"}> {/* Adjust layout as needed */}
      {/*<div>*/}
      {/*  {errorLogs.map(log => (*/}
      {/*    <div key={log.id}>*/}
      {/*      <p><strong>Timestamp:</strong> {log.createdAt ? new Date(log.createdAt).toLocaleString() : 'N/A'}</p>*/}
      {/*      <p><strong>Message:</strong> {log.message || 'No message'}</p>*/}
      {/*      <hr />*/}
      {/*    </div>*/}
      {/*  ))}*/}
      {/*</div>*/}
      {/*<div> /!* New table for users *!/*/}
      {/*  <table>*/}
      {/*    <thead>*/}
      {/*      <tr>*/}
      {/*        <th>UserId</th>*/}
      {/*        <th>Email</th>*/}
      {/*        <th>UserTypeId</th> /!* Adjust according to how you want to display user types *!/*/}
      {/*      </tr>*/}
      {/*    </thead>*/}
      {/*    <tbody>*/}
      {/*      {users.map((user) => (*/}
      {/*        <tr key={user.id}>*/}
      {/*          <td>{user.id}</td>*/}
      {/*          <td>{user.email}</td>*/}
      {/*          <td>{user.typeId}</td> /!* Assuming typeId is available, adjust as needed *!/*/}
      {/*        </tr>*/}
      {/*      ))}*/}
      {/*    </tbody>*/}
      {/*  </table>*/}
      {/*</div>*/}
        <div className="floatingDiv col-md-9 col-12">
          <h3> Error Logs </h3>
          <hr/>
          <DataTable
              pagination
              columns={errorHeaders}
              data={errorLogs}
              striped
          />
        </div>
        <div className="col-md-3 col-12"></div>
        <div className="floatingDiv col-md-6 col-12 mt-3">
            <h3> Users </h3>
            <hr/>
            <DataTable
                pagination
                columns={userHeaders}
                data={users}
                striped
            />
        </div>
    </div>
  );
};

export default AdminView;
