import React, { useEffect, useState } from 'react';
import ErrorLog from "../Interfaces/ErrorLog";
import User from "../Interfaces/User"; // Import the User interface
import { getErrorLogs, getUsers } from "../Services/ErrorLogService"; // Ensure getUsers is imported
import DataTable from 'react-data-table-component';

const AdminView: React.FC = () => {
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [users, setUsers] = useState<User[]>([]); // Add state for users

  useEffect(() => {
    const fetchData = async () => {
      let logs = await getErrorLogs();
  
      // Sort logs by createdAt in descending order
      logs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  
      setErrorLogs(logs);
  
      const users = await getUsers(); // Fetch users
      setUsers(users);
    };
  
    fetchData();
  }, []);

  const errorHeaders = [
      {
        name: 'Id',
        selector: (row: ErrorLog) => row.id,
        sortable: true,
        width: "100px",
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
